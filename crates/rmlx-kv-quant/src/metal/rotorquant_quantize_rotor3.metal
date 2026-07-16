
uint gid   = thread_position_in_grid.x;
uint n_grp = n_groups[0];
uint hd    = head_dim[0];
uint token = gid / n_grp;
uint grp   = gid % n_grp;

// ── Per-token L2 norm (recomputed per group, no shared memory) ────────────
float norm_sq = 0.0f;
for (uint i = 0u; i < hd; i++) {
    float vi = inp[token * hd + i];
    norm_sq += vi * vi;
}
float norm = sqrt(norm_sq);
if (norm < 1e-8f)
    norm = 1e-8f;
norms_out[gid] = norm;

// ── Load 3 grade-1 components with tail-pad (head_dim % 3 may be != 0) ────
uint grp_start = grp * ROTOR3_GS;
float v1       = (grp_start + 0u < hd) ? (inp[token * hd + grp_start + 0u] / norm) : 0.0f;
float v2       = (grp_start + 1u < hd) ? (inp[token * hd + grp_start + 1u] / norm) : 0.0f;
float v3       = (grp_start + 2u < hd) ? (inp[token * hd + grp_start + 2u] / norm) : 0.0f;

// ── Load rotor [s, b12, b13, b23] ─────────────────────────────────────────
uint r_base = grp * 4u;
float s     = rotors_in[r_base + 0u];
float b12   = rotors_in[r_base + 1u];
float b13   = rotors_in[r_base + 2u];
float b23   = rotors_in[r_base + 3u];

// ── Apply 3×3 SO(3) rotation matrix M(R) (derived from R * mv * R̃) ──────
float s2    = s * s;
float b12_2 = b12 * b12;
float b13_2 = b13 * b13;
float b23_2 = b23 * b23;

float R1 =
    (s2 - b12_2 - b13_2 + b23_2) * v1 + (2.0f * s * b12 - 2.0f * b13 * b23) * v2 + (2.0f * s * b13 + 2.0f * b12 * b23) * v3;
float R2 =
    (-2.0f * s * b12 - 2.0f * b23 * b13) * v1 + (s2 - b12_2 - b23_2 + b13_2) * v2 + (2.0f * s * b23 - 2.0f * b12 * b13) * v3;
float R3 =
    (-2.0f * s * b13 + 2.0f * b23 * b12) * v1 + (-2.0f * s * b23 - 2.0f * b13 * b12) * v2 + (s2 - b13_2 - b23_2 + b12_2) * v3;

// ── 8-component MV: [0, R1, R2, R3, 0, 0, 0, 0] ──────────────────────────
float rots[8];
rots[0] = 0.0f;
rots[1] = R1;
rots[2] = R2;
rots[3] = R3;
rots[4] = 0.0f;
rots[5] = 0.0f;
rots[6] = 0.0f;
rots[7] = 0.0f;

// ── Per-group scale = max|R_i| / CB_MAX ──────────────────────────────────
float abs_max   = max(max(abs(R1), abs(R2)), abs(R3));
float scale     = (abs_max < 1e-12f) ? 1e-12f : (abs_max / ROTOR3_CB_MAX);
scales_out[gid] = scale;

// ── 3-bit quantize all 8 components and pack into 1 u32 via atomic OR ────
uint code_word = gid * ROTOR3_WPG; // = gid (WPG = 1)
for (uint e = 0u; e < ROTOR3_MV; e++) {
    float norm_val = (scale > 0.0f) ? (rots[e] / scale) : 0.0f;
    uint idx       = 0u;
    for (uint bi = 0u; bi < 7u; bi++) {
        if (norm_val > ROTOR3_BOUNDS[bi])
            idx++;
    }
    uint shift = e * 3u;
    atomic_fetch_or_explicit((device atomic_uint *)&codes_out[code_word],
                             (idx & 0x7u) << shift,
                             memory_order_relaxed);
}
