// iso4 dequantize: one thread per (token, group) pair.
uint gid        = thread_position_in_grid.x;
uint n_groups_u = n_groups[0];
uint token      = gid / n_groups_u;
uint grp        = gid % n_groups_u;
uint hd         = n_groups_u * ISO4_GS;

float scale = scales_in[gid];
float norm  = norms_in[gid];

uint code_word = gid * ISO4_WPG;
uint base_out  = token * hd + grp * ISO4_GS;

// ── Unpack, dequantize, inverse-rotate, rescale ───────────────────────────
float rots[4];
for (uint e = 0u; e < ISO4_GS; e++) {
    uint word  = code_word + e / ISO4_VPW;
    uint shift = (e % ISO4_VPW) * 4u;
    uint idx   = (codes_in[word] >> shift) & 0xFu;
    rots[e]    = ISO4_CB[idx] * scale;
}

float rw = rots[0];
float rx = rots[1];
float ry = rots[2];
float rz = rots[3];

// Inverse rotation: q̄_L * r — Hamilton product with conjugate.
float vw = ISO4_QW * rw - ISO4_CX * rx - ISO4_CY * ry - ISO4_CZ * rz;
float vx = ISO4_QW * rx + ISO4_CX * rw + ISO4_CY * rz - ISO4_CZ * ry;
float vy = ISO4_QW * ry - ISO4_CX * rz + ISO4_CY * rw + ISO4_CZ * rx;
float vz = ISO4_QW * rz + ISO4_CX * ry - ISO4_CY * rx + ISO4_CZ * rw;

out[base_out]     = vw * norm;
out[base_out + 1] = vx * norm;
out[base_out + 2] = vy * norm;
out[base_out + 3] = vz * norm;
