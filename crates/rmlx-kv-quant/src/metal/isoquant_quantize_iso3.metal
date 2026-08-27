
uint gid        = thread_position_in_grid.x;
uint n_groups_u = n_groups[0];
uint token      = gid / n_groups_u;
uint grp        = gid % n_groups_u;
uint hd         = n_groups_u * ISO3_GS; // head_dim

// ── Compute per-token L2 norm (recomputed per group — hook impl) ──────────
float norm_sq = 0.0f;
for (uint i = 0u; i < hd; i++) {
    float vi = inp[token * hd + i];
    norm_sq += vi * vi;
}
// Rounded to the stored sideband precision before it is used: the decode
// multiplies by the *stored* norm, so quantizing against a finer one bakes
// in an error the store cannot represent.
float norm = float(bfloat(max(sqrt(norm_sq), 1e-8f)));

// The store's element type is the sideband dtype; MSL requires the
// narrowing to be written out (bfloat has no implicit conversion from
// float, unlike half).
norms_out[gid] = bfloat(norm); // store per group slot

// ── Load 4 elements, normalise, apply Hamilton product r = q_L * v ────────
uint base = token * hd + grp * ISO3_GS;
float vw  = inp[base] / norm;
float vx  = inp[base + 1] / norm;
float vy  = inp[base + 2] / norm;
float vz  = inp[base + 3] / norm;

// Hamilton product: r = q_L * v, [w,x,y,z] convention.
float rw = ISO3_QW * vw - ISO3_QX * vx - ISO3_QY * vy - ISO3_QZ * vz;
float rx = ISO3_QW * vx + ISO3_QX * vw + ISO3_QY * vz - ISO3_QZ * vy;
float ry = ISO3_QW * vy - ISO3_QX * vz + ISO3_QY * vw + ISO3_QZ * vx;
float rz = ISO3_QW * vz + ISO3_QX * vy - ISO3_QY * vx + ISO3_QZ * vw;

// ── Per-group scale ────────────────────────────────────────────────────────
float abs_max   = max(max(abs(rw), abs(rx)), max(abs(ry), abs(rz)));
float scale     = float(bfloat((abs_max < 1e-12f) ? 1e-12f : (abs_max / ISO3_CB_MAX)));
scales_out[gid] = bfloat(scale);

// ── 3-bit quantize (codebook lookup) and pack via atomic OR ───────────────
// ISO3_GS=4 elements → 1 u32 per group (4*3=12 bits ≤ 30).
uint code_word = gid * ISO3_WPG; // = gid * 1 for ISO3_GS=4
float rots[4];
rots[0] = rw;
rots[1] = rx;
rots[2] = ry;
rots[3] = rz;

for (uint e = 0u; e < ISO3_GS; e++) {
    float norm_val = (scale > 0.0f) ? (rots[e] / scale) : 0.0f;
    uint idx       = 0u;
    for (uint bi = 0u; bi < 7u; bi++) {
        if (norm_val > ISO3_BOUNDS[bi])
            idx++;
    }
    uint word  = code_word + e / ISO3_VPW;
    uint shift = (e % ISO3_VPW) * 3u;
    atomic_fetch_or_explicit((device atomic_uint *)&codes_out[word],
                             (idx & 0x7u) << shift,
                             memory_order_relaxed);
}
