//! Tests for [`QuantPlanarV`].
//!
//! `QuantPlanarV` accumulates one `PlanarBlocks` per `append` and reorders them
//! back to head-major on `dequantize_choice`, the same shape as every other
//! block-list store. It was the one converted store with no sibling test file,
//! so its call into `seq_layout::transpose_chunked_seq_heads` had no direct
//! coverage at all — including the `B == 1` equivalence half, where an argument
//! transposed at the call site (`kv_h` / `d` swapped, or a truncated block list)
//! would ship green.

use super::QuantPlanarV;
use rmlx_mlx::{zeros, Device, Dtype};

fn new_planar_v(b: usize, kv_h: usize, d: usize, bits: u8, max_seq: i32) -> QuantPlanarV {
    QuantPlanarV {
        blocks: Vec::new(),
        gpu_codes_buf: None,
        gpu_scales_buf: None,
        gpu_rotations_buf: None,
        gpu_codes_words_per_step: 0,
        gpu_scales_per_step: 0,
        gpu_rotations_words_per_step: 0,
        gpu_capacity: 0,
        shape: vec![b as i32, kv_h as i32, 0, d as i32],
        max_seq,
        bits,
    }
}

/// Two appends must decode exactly like one append of the same tokens, at
/// `B > 1` as well as `B == 1`.
///
/// Each block covers `[B, S_block, kv_h, D]`, so the concatenation of two blocks
/// is not one `[B, S_total, kv_h, D]` run — reading it as one maps the second
/// block's batch-0 rows onto batch-1 sequence slots. The single-append store
/// holds exactly one block and therefore concatenates nothing, which is what
/// makes it the oracle here.
///
/// Mutation check: put `seq_layout::transpose_seq_heads` over the whole
/// concatenation back in `QuantPlanarV::dequantize_choice` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
#[test]
fn quant_planar_v_two_block_decode_matches_one_block_at_b_gt_1() {
    for (b, kv_h) in [(1_usize, 1_usize), (1, 2), (2, 1), (2, 2)] {
        let head_dim = 32_usize;
        let (n0, n1) = (2_usize, 3_usize);
        let max_seq = 512_i32;
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];
        let dummy = |n: usize| zeros(&shape(n), Dtype::F32, Device::Cpu).expect("dummy array");
        let cpu_dequant = |st: &QuantPlanarV| {
            st.dequantize_choice(Device::Cpu, Dtype::F32)
                .expect("cpu dequant")
                .0
        };

        let mut one = new_planar_v(b, kv_h, head_dim, 4, max_seq);
        one.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0 + n1, head_dim),
            &shape(n0 + n1),
            &dummy(n0 + n1),
            Device::Cpu,
            max_seq,
        )
        .expect("single append");
        let oracle = cpu_dequant(&one);

        let mut two = new_planar_v(b, kv_h, head_dim, 4, max_seq);
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0, head_dim),
            &shape(n0),
            &dummy(n0),
            Device::Cpu,
            max_seq,
        )
        .expect("append chunk 0");
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, n0, n1, head_dim),
            &shape(n1),
            &dummy(n1),
            Device::Cpu,
            max_seq,
        )
        .expect("append chunk 1");
        assert_eq!(
            two.blocks.len(),
            2,
            "the two-append store really holds 2 blocks"
        );
        let got = cpu_dequant(&two);

        assert_eq!(
            got, oracle,
            "two-block decode must equal the one-block oracle at b={b} kv_h={kv_h}"
        );
    }
}
