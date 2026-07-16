
uint gid      = thread_position_in_grid.x;
uint group_id = gid / 128u;
uint elem     = gid % 128u;

uint word_idx = group_id * 32u + (elem / 4u);
uint byte_pos = elem & 3u;
uint word     = codes[word_idx];
uint raw_byte = (word >> (byte_pos * 8u)) & 0xFFu;

int code = (int)raw_byte;
if (code & 0x80) {
    code -= 256;
}

float scale = scales[group_id];
out[gid]    = static_cast<OutT>(scale * (float)code);
