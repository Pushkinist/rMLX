
uint pair_id   = thread_position_in_grid.x;
uint group_id  = pair_id / PAIRS_PER_GROUP_C;
uint pair_in_g = pair_id % PAIRS_PER_GROUP_C;

// ── Unpack rotation index ─────────────────────────────────────────────────
uint rot_word  = group_id * 2u + (pair_in_g / 8u);
uint rot_shift = (pair_in_g % 8u) * 4u;
uint rot_idx   = (rot32[rot_word] >> rot_shift) & 0xFu;

// ── Unpack code indices ───────────────────────────────────────────────────
uint elem_a  = pair_in_g * 2u;
uint elem_b  = pair_in_g * 2u + 1u;
uint word_a  = group_id * 4u + (elem_a / 8u);
uint word_b  = group_id * 4u + (elem_b / 8u);
uint shift_a = (elem_a % 8u) * 4u;
uint shift_b = (elem_b % 8u) * 4u;
uint idx_a   = (codes[word_a] >> shift_a) & 0xFu;
uint idx_b   = (codes[word_b] >> shift_b) & 0xFu;

// ── Dequantize in rotated space ───────────────────────────────────────────
float scale = scales[pair_id];
float ya    = CB[idx_a] * scale;
float yb    = CB[idx_b] * scale;

// ── Apply R^T: [[c, s], [-s, c]] (entry = [c, -s, s, c]) ─────────────────
float c     = ROT_CB[rot_idx][0];
float neg_s = ROT_CB[rot_idx][1];
float sv    = ROT_CB[rot_idx][2];
float c2    = ROT_CB[rot_idx][3];

// R^T col0 = [c, -s]^T rotated back: a = c*ya + s*yb, b = -s*ya + c*yb
float ao = c * ya + sv * yb;
float bo = neg_s * ya + c2 * yb;

out[pair_id * 2u]      = ao;
out[pair_id * 2u + 1u] = bo;
