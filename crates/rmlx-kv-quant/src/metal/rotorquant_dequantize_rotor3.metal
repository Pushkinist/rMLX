
uint gid   = thread_position_in_grid.x;
uint n_grp = n_groups[0];
uint hd    = head_dim[0];
uint token = gid / n_grp;
uint grp   = gid % n_grp;

float scale = scales_in[gid];
float norm  = norms_in[gid];

// ── Read the three stored grade-1 codes out of the dense plane ───────────
uint row_base = token * cp_row_words(n_grp);
cp_group_t g  = cp_read_group(codes_in, row_base, grp);
float c1      = ROTOR3_CB[g.x] * scale;
float c2      = ROTOR3_CB[g.y] * scale;
float c3      = ROTOR3_CB[g.z] * scale;

// ── Load rotor and apply inverse rotation M(R)^T = M(R̃) ─────────────────
uint r_base = grp * 4u;
float s     = rotors_in[r_base + 0u];
float b12   = rotors_in[r_base + 1u];
float b13   = rotors_in[r_base + 2u];
float b23   = rotors_in[r_base + 3u];

float s2    = s * s;
float b12_2 = b12 * b12;
float b13_2 = b13 * b13;
float b23_2 = b23 * b23;

// Transpose of M(R) — equivalent to M(R̃) where R̃ flips bivector signs.
float v1 =
    (s2 - b12_2 - b13_2 + b23_2) * c1 + (-2.0f * s * b12 - 2.0f * b23 * b13) * c2 + (-2.0f * s * b13 + 2.0f * b23 * b12) * c3;
float v2 =
    (2.0f * s * b12 - 2.0f * b13 * b23) * c1 + (s2 - b12_2 - b23_2 + b13_2) * c2 + (-2.0f * s * b23 - 2.0f * b13 * b12) * c3;
float v3 =
    (2.0f * s * b13 + 2.0f * b12 * b23) * c1 + (2.0f * s * b23 - 2.0f * b12 * b13) * c2 + (s2 - b13_2 - b23_2 + b12_2) * c3;

// ── Write grade-1 components × per-token norm, mask tail-pad ─────────────
uint grp_start = grp * ROTOR3_GS;
if (grp_start + 0u < hd)
    out[token * hd + grp_start + 0u] = v1 * norm;
if (grp_start + 1u < hd)
    out[token * hd + grp_start + 1u] = v2 * norm;
if (grp_start + 2u < hd)
    out[token * hd + grp_start + 2u] = v3 * norm;
