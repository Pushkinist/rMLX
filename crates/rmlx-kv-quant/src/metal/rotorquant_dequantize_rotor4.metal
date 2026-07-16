
uint gid   = thread_position_in_grid.x;
uint n_grp = n_groups[0];
uint hd    = head_dim[0];
uint token = gid / n_grp;
uint grp   = gid % n_grp;

float scale = scales_in[gid];
float norm  = norms_in[gid];

uint word = codes_in[gid * ROTOR4_WPG];
float rots[8];
for (uint e = 0u; e < ROTOR4_MV; e++) {
    uint shift = e * 4u;
    uint idx   = (word >> shift) & 0xFu;
    rots[e]    = ROTOR4_CB[idx] * scale;
}

float c1 = rots[1];
float c2 = rots[2];
float c3 = rots[3];

uint r_base = grp * 4u;
float s     = rotors_in[r_base + 0u];
float b12   = rotors_in[r_base + 1u];
float b13   = rotors_in[r_base + 2u];
float b23   = rotors_in[r_base + 3u];

float s2    = s * s;
float b12_2 = b12 * b12;
float b13_2 = b13 * b13;
float b23_2 = b23 * b23;

float v1 =
    (s2 - b12_2 - b13_2 + b23_2) * c1 + (-2.0f * s * b12 - 2.0f * b23 * b13) * c2 + (-2.0f * s * b13 + 2.0f * b23 * b12) * c3;
float v2 =
    (2.0f * s * b12 - 2.0f * b13 * b23) * c1 + (s2 - b12_2 - b23_2 + b13_2) * c2 + (-2.0f * s * b23 - 2.0f * b13 * b12) * c3;
float v3 =
    (2.0f * s * b13 + 2.0f * b12 * b23) * c1 + (2.0f * s * b23 - 2.0f * b12 * b13) * c2 + (s2 - b13_2 - b23_2 + b12_2) * c3;

uint grp_start = grp * ROTOR4_GS;
if (grp_start + 0u < hd)
    out[token * hd + grp_start + 0u] = v1 * norm;
if (grp_start + 1u < hd)
    out[token * hd + grp_start + 1u] = v2 * norm;
if (grp_start + 2u < hd)
    out[token * hd + grp_start + 2u] = v3 * norm;
