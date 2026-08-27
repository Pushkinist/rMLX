// iso4 quantize: one thread per (token, group) pair. Grid = n_tokens * n_groups.
//
// Outputs:
//   codes_out  : u32 [n_tokens * n_groups * ISO4_WPG] — 4-bit codes, 8/u32
//   scales_out : bf16 [n_tokens * n_groups]           — per-group scale
//   norms_out  : bf16 [n_tokens * n_groups]           — per-group L2 norm slot
uint gid        = thread_position_in_grid.x;
uint n_groups_u = n_groups[0];
uint token      = gid / n_groups_u;
uint grp        = gid % n_groups_u;
uint hd         = n_groups_u * ISO4_GS; // head_dim

// ── Compute per-token L2 norm (recomputed per group ───────────────────────
//     redundant by design — one thread per (token, group), no shared memory for n_groups norms) ──
float norm_sq = 0.0f;
for (uint i = 0u; i < hd; i++) {
    float vi = inp[token * hd + i];
    norm_sq += vi * vi;
}
// Rounded to the stored sideband precision before use — see
// isoquant_quantize_iso3.metal.
float norm = float(bfloat(max(sqrt(norm_sq), 1e-8f)));

// Narrowing to the sideband dtype is explicit: MSL has no implicit
// float -> bfloat conversion.
norms_out[gid] = bfloat(norm);

// ── Load 4 elements, normalise, apply Hamilton product r = q_L * v ────────
uint base = token * hd + grp * ISO4_GS;
float vw  = inp[base] / norm;
float vx  = inp[base + 1] / norm;
float vy  = inp[base + 2] / norm;
float vz  = inp[base + 3] / norm;

float rw = ISO4_QW * vw - ISO4_QX * vx - ISO4_QY * vy - ISO4_QZ * vz;
float rx = ISO4_QW * vx + ISO4_QX * vw + ISO4_QY * vz - ISO4_QZ * vy;
float ry = ISO4_QW * vy - ISO4_QX * vz + ISO4_QY * vw + ISO4_QZ * vx;
float rz = ISO4_QW * vz + ISO4_QX * vy - ISO4_QY * vx + ISO4_QZ * vw;

// ── Per-group scale ───────────────────────────────────────────────────────
float abs_max   = max(max(abs(rw), abs(rx)), max(abs(ry), abs(rz)));
float scale     = float(bfloat((abs_max < 1e-12f) ? 1e-12f : (abs_max / ISO4_CB_MAX)));
scales_out[gid] = bfloat(scale);

// ── 4-bit quantize (codebook lookup) and pack via atomic OR ───────────────
// ISO4_GS=4 elements → 1 u32 per group (4*4=16 bits ≤ 32).
uint code_word = gid * ISO4_WPG; // = gid * 1 for ISO4_GS=4
float rots[4];
rots[0] = rw;
rots[1] = rx;
rots[2] = ry;
rots[3] = rz;

for (uint e = 0u; e < ISO4_GS; e++) {
    float norm_val = (scale > 0.0f) ? (rots[e] / scale) : 0.0f;
    uint idx       = 0u;
    for (uint bi = 0u; bi < 15u; bi++) {
        if (norm_val > ISO4_BOUNDS[bi])
            idx++;
    }
    uint word  = code_word + e / ISO4_VPW;
    uint shift = (e % ISO4_VPW) * 4u;
    atomic_fetch_or_explicit((device atomic_uint *)&codes_out[word],
                             (idx & 0xFu) << shift,
                             memory_order_relaxed);
}
