
uint group_id = threadgroup_position_in_grid.x;
uint elem     = thread_position_in_threadgroup.x;

threadgroup uint words[3];
if (elem < 3u) {
    words[elem] = codes[group_id * 3u + elem];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Bits [elem*3, elem*3+3) of the concatenated 96-bit stream.
uint bit_off  = elem * 3u;
uint word0_id = bit_off / 32u; // 0, 1 or 2
uint shift0   = bit_off - word0_id * 32u;
ulong window  = (ulong)words[word0_id];
if (word0_id + 1u < 3u) {
    window |= ((ulong)words[word0_id + 1u]) << 32;
}
uint idx = (uint)((window >> shift0) & 0x7ul);

float scale                = scales[group_id];
float v                    = CB3[idx] * scale;
out[group_id * 32u + elem] = static_cast<OutT>(v);
