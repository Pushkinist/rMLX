
 // gid = word index (0 .. N_groups*4-1).
    uint gid      = thread_position_in_grid.x;
    uint group_id = gid / 4u;   // which group of 32 elements
    uint word_off = gid % 4u;   // word within group (0..3), 8 elements each

    uint word  = codes[gid];
    float scale = scales[group_id];

 // Base output index for this word's 8 elements.
    uint base_out = group_id * 32u + word_off * 8u;

 // Dual-LUT: decode each of the 4 bytes in the word, 2 nibbles per byte.
 // Equivalent to a 256-entry half2 LUT: CB_LUT[b] = (CB[b & 0xF], CB[b >> 4]).
 // CB is in Metal constant memory — repeated lookups hit L1 after first use.
    for (uint byte_idx = 0u; byte_idx < 4u; byte_idx++) {
        uint b   = (word >> (byte_idx * 8u)) & 0xFFu;
        uint lo  = b & 0xFu;
        uint hi  = b >> 4u;
        float v0 = CB[lo] * scale;
        float v1 = CB[hi] * scale;
        out[base_out + byte_idx * 2u    ] = static_cast<OutT>(v0);
        out[base_out + byte_idx * 2u + 1] = static_cast<OutT>(v1);
    }
