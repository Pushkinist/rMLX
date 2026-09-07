
uint gid   = thread_position_in_grid.x;
uint n_grp = n_groups[0];
uint hd    = head_dim[0];
uint token = gid / n_grp;
uint grp   = gid % n_grp;

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

uint grp_start = grp * ROTOR4_GS;
float v1       = (grp_start + 0u < hd) ? (inp[token * hd + grp_start + 0u] / norm) : 0.0f;
float v2       = (grp_start + 1u < hd) ? (inp[token * hd + grp_start + 1u] / norm) : 0.0f;
float v3       = (grp_start + 2u < hd) ? (inp[token * hd + grp_start + 2u] / norm) : 0.0f;

uint r_base = grp * 4u;
float s     = rotors_in[r_base + 0u];
float b12   = rotors_in[r_base + 1u];
float b13   = rotors_in[r_base + 2u];
float b23   = rotors_in[r_base + 3u];

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

float abs_max   = max(max(abs(R1), abs(R2)), abs(R3));
float scale     = float(bfloat((abs_max < 1e-12f) ? 1e-12f : (abs_max / ROTOR4_CB_MAX)));
scales_out[gid] = bfloat(scale);

// ── Quantize the three grade-1 components into the dense code plane ──────
// The other five multivector components are algebraically zero — the rotor
// sandwich preserves grade — so they are not stored.
float rots[CP_CODES_PER_GROUP];
rots[0] = R1;
rots[1] = R2;
rots[2] = R3;

uint row_base = token * cp_row_words(n_grp);
for (uint e = 0u; e < CP_CODES_PER_GROUP; e++) {
    float norm_val = (scale > 0.0f) ? (rots[e] / scale) : 0.0f;
    uint idx       = 0u;
    for (uint bi = 0u; bi < 15u; bi++) {
        if (norm_val > ROTOR4_BOUNDS[bi])
            idx++;
    }
    cp_write_code(codes_out, row_base, grp * CP_CODES_PER_GROUP + e, idx);
}
