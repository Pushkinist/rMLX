
uint group_id = threadgroup_position_in_grid.x;
uint elem     = thread_position_in_threadgroup.x;

threadgroup uint words[2];
if (elem < 2u) {
    words[elem] = codes[group_id * 2u + elem];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Bits [elem*2, elem*2+2) of the concatenated 64-bit stream.
uint bit_off  = elem * 2u;
uint word0_id = bit_off / 32u; // 0 or 1
uint shift0   = bit_off - word0_id * 32u;
ulong window  = (ulong)words[word0_id];
if (word0_id + 1u < 2u) {
    window |= ((ulong)words[word0_id + 1u]) << 32;
}
uint idx = (uint)((window >> shift0) & 0x3ul);

float scale                = scales[group_id];
float v                    = CB2[idx] * scale;
out[group_id * 32u + elem] = static_cast<OutT>(v);
