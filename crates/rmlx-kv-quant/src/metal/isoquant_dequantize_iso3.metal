
uint gid        = thread_position_in_grid.x;
uint n_groups_u = n_groups[0];
uint token      = gid / n_groups_u;
uint grp        = gid % n_groups_u;
uint hd         = n_groups_u * ISO3_GS;

float scale = scales_in[gid];
float norm  = norms_in[gid];

uint code_word = gid * ISO3_WPG;
uint base_out  = token * hd + grp * ISO3_GS;

// ── Unpack, dequantize, inverse-rotate, rescale ───────────────────────────
float rots[4];
for (uint e = 0u; e < ISO3_GS; e++) {
    uint word  = code_word + e / ISO3_VPW;
    uint shift = (e % ISO3_VPW) * 3u;
    uint idx   = (codes_in[word] >> shift) & 0x7u;
    rots[e]    = ISO3_CB[idx] * scale;
}

float rw = rots[0];
float rx = rots[1];
float ry = rots[2];
float rz = rots[3];

// Inverse rotation: q̄_L * r — Hamilton product with conjugate.
// q̄_L = (ISO3_QW, ISO3_CX, ISO3_CY, ISO3_CZ).
float vw = ISO3_QW * rw - ISO3_CX * rx - ISO3_CY * ry - ISO3_CZ * rz;
float vx = ISO3_QW * rx + ISO3_CX * rw + ISO3_CY * rz - ISO3_CZ * ry;
float vy = ISO3_QW * ry - ISO3_CX * rz + ISO3_CY * rw + ISO3_CZ * rx;
float vz = ISO3_QW * rz + ISO3_CX * ry - ISO3_CY * rx + ISO3_CZ * rw;

out[base_out]     = vw * norm;
out[base_out + 1] = vx * norm;
out[base_out + 2] = vy * norm;
out[base_out + 3] = vz * norm;
