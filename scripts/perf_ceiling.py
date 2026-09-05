#!/usr/bin/env python3
"""Theoretical decode/prefill ceiling calculator for rMLX on Apple Silicon.

Pure static analysis: reads a model snapshot's `config.json` + safetensors
headers and a KV codec name, and prints the bandwidth-bound decode ceiling and
a compute-bound prefill projection per context length. Never launches a model,
never touches the GPU, never takes the MLX claim.

Usage:
    scripts/perf_ceiling.py --model <snapshot> --kv-quant k8v8 \
        --ctx 4096 --ctx 8192 --ctx 32768
    scripts/perf_ceiling.py --model <snapshot> --kv-quant none --ctx 4096 --json

--------------------------------------------------------------------------
Provenance of every constant (verify before trusting; see "rule zero")
--------------------------------------------------------------------------

HOST_BW_BYTES_PER_S = 614e9
    Measured unified-memory bandwidth ceiling for the M5 Max 128 GB dev host.
    Source: docs/PERF_BASELINE.md:101 ("Hardware: M5 Max, bandwidth ceiling
    614 GB/s"). Override with --bandwidth-gbs for another host.

KV byte accounting mirrors the engine, not a re-invention:
    crates/rmlx-kv-quant/src/quant.rs  KvQuant::estimated_resident_bytes_per_layer
    crates/rmlx-kv-quant/src/quant.rs  KvQuant::side_stores / SideStore /
        packed_side_bytes -- the per-side store layout and its byte cadence.
        `_PLANAR` / `_ISO` / `_ISO_RING` / `_ROTOR` / `_K_FAMILY` / `_V_FAMILY`
        below are that split transcribed.
    crates/rmlx-kv-quant/src/quant.rs  KvQuant::approx_code_bits
    crates/rmlx-kv-quant/src/quant.rs  KvQuant::feeds_bf16_k_at_decode
    crates/rmlx-kv-quant/src/quant.rs  KvQuant::feeds_bf16_v_at_decode
        -- both take the stack's `shares_kv`, which moves Mixed / RotK by two
        whole bf16 mirrors and moves nothing else.
    crates/rmlx-kv-quant/src/quant.rs      KvQuant::decode_reads_packed_store /
        materialises_packed_store -- a codec that reads no store gets none
        built, so its resident KV is the two bf16 mirrors and nothing more.
        `_DECODE_READS_PACKED_STORE` below is that match transcribed arm for
        arm.
    crates/rmlx-models/src/kv_cache/mod.rs:229  kv_codec_net_saving_total
        (per-layer loop: windowed layers clamp seq to the window and are
        always bf16; global layers take the codec formula)
    crates/rmlx-kv-quant/src/kvcache/core.rs:346  with_quant_max_seq_window
        ("uses the RotatingKvCache code path ... regardless of the `quant`
        flag") -- an SWA layer is bf16 at `sliding_window` tokens, always.
    crates/rmlx-kv-quant/src/rotating.rs:7  ring is `[B, kv_h, max_size, D]`
    crates/rmlx-models/src/kv_cache/mod.rs  kv_quant_for_layer / boundary_floor
        (first HEAD_N=2 and last TAIL_N=8 layers take an 8-bit quality floor for
        every base codec that quantizes a side; a base whose widths are
        parameters is raised inside its own family, one whose width is baked
        into the variant falls back to K8V8, and a target that still mirrors
        both axes is diverted to K8V8; `none` is exempt)
    crates/rmlx-kv-quant/src/kvcache/update.rs:7982  next_pow2_seq
        (ring capacity = min(next_pow2(needed), --max-ctx ceiling))

Every one of those is a SECOND producer of arithmetic the engine already owns, so
`make check-kv-byte-model-parity` diffs the two models per codec, per topology
and per head dimension, over the codec sweep the engine emits. Move the engine
and this script goes red until it follows.

PREFILL_ACHIEVED_FLOPS has NO hard-coded default on purpose. A single
"achieved GEMM throughput" constant is not defensible on this host: the
recorded prefill_tps rows in .rmlx/metrics/runs.db imply between ~8 and
~57 TFLOP/s depending on the model, a 7x spread (MoE gather_qmm prefill and
2-bit dequant prefill are far from the dense mxfp8 rate). The script therefore
anchors per model on a real measured row from runs.db and only uses the FLOP
model to extrapolate across context length. With no qualifying anchor it
reports `prefill_ceiling_tps = null` rather than guessing.

PREFILL_ANCHOR_MIN_TS = "2026-06-20"
    Recorded prefill_tps for Qwen3.6-35B-A3B steps from ~600 to ~3800 TPS on
    2026-06-20 and stays there; Bonsai (+25%) and gemma-4-e4b (+10%) move in
    the same window by much less, so this is an arch-level prefill change, not
    a host-level one. Rows older than this are excluded as a different regime.
    NOT a verified NAX-linkage boundary -- the cause was not established.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Host constants ───────────────────────────────────────────────────────────

HOST_BW_BYTES_PER_S = 614e9  # docs/PERF_BASELINE.md:101

# Anchor-selection policy for the prefill projection (see module docstring).
PREFILL_ACHIEVED_FLOPS: float | None = None
PREFILL_ANCHOR_MIN_TS = "2026-06-20"
PREFILL_ANCHOR_MIN_N = 3
PREFILL_ANCHOR_MAX_SPREAD = 2.0  # reject an anchor cell whose max/min exceeds this

# crates/rmlx-models/src/kv_cache/mod.rs:82,92
LAYER_ADAPTIVE_TAIL_N = 8
LAYER_ADAPTIVE_HEAD_N = 2

BF16 = 2  # bytes per bf16 element

# Bytes one scale -- or one norm -- occupies in a GPU-resident iso/rotor ring.
# Mirror of `SIDEBAND_BYTES` (quant.rs), which reads `KV_SIDEBAND_DTYPE`
# (storage/quant_k_gpu_ring.rs). It is bf16, not f32: past the dense code plane
# the sideband is the dominant term -- 4.125 of iso3's 7.125 stored bits per
# value at head_dim 128 -- so modelling it at 4 bytes would price the family
# half again over.
_RING_SIDEBAND_BYTES = 2

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


# ── KvQuant model (mirror of crates/rmlx-kv-quant/src/quant.rs) ──────────────

@dataclass(frozen=True)
class Codec:
    """Parsed KvQuant. `name` is the canonical Display string."""

    name: str
    kind: str          # variant tag matching the Rust enum name
    k_bits: int
    v_bits: int
    k_group: int = 0   # affine group size; 0 = not an affine-grouped codec
    v_group: int = 0


_SIMPLE = {
    # canonical string -> (variant tag, k_bits, v_bits)  [quant.rs approx_code_bits]
    "none": ("None", 16, 16),
    "bf16": ("None", 16, 16),
    "f16": ("None", 16, 16),
    "k8v8": ("K8V8", 8, 8),
    "k8v4": ("K8V4", 8, 4),
    "planar": ("Planar", 8, 4),
    "planar3": ("Planar3", 8, 3),
    "planar_k": ("PlanarK", 4, 16),
    "k8vturbo3": ("K8VTurbo3", 8, 3),
    "k8vturbo3tcq": ("K8VTurbo3Tcq", 8, 3),
    "k8vturbo2": ("K8VTurbo2", 8, 2),
    "k8vturbo2tcq": ("K8VTurbo2Tcq", 8, 2),
    "tsym3": ("TurboSym3", 3, 3),
    "tsym4": ("TurboSym4", 4, 4),
    "iso3": ("Iso3", 8, 3),
    "iso4": ("Iso4", 8, 4),
    "iso3_sym": ("Iso3Sym", 3, 3),
    "iso4_sym": ("Iso4Sym", 4, 4),
    "k_iso3": ("IsoKOnly3", 3, 16),
    "k_iso4": ("IsoKOnly4", 4, 16),
    "rotor3": ("Rotor3", 8, 3),
    "rotor4": ("Rotor4", 8, 4),
    "rotor3_sym": ("Rotor3Sym", 3, 3),
    "rotor4_sym": ("Rotor4Sym", 4, 4),
    "k_rotor3": ("RotorKOnly3", 3, 16),
    "k_rotor4": ("RotorKOnly4", 4, 16),
}

# Codecs whose decode dispatches quantized_matmul over the packed affine store
# (`uses_mixed_path`, quant.rs:458-460 -> `mixed_quantized_sdpa`).
#
# PROVEN BY GPU CAPTURE, not by the feeds_bf16_* predicate: a decode window under
# `mixed_k8g64_v4g64` references `affine_qmv_quad_*_gs_64_b_8` (QK over packed K)
# and `affine_qvm_split_k_*_gs_64_b_4` (PV over packed V) on both a kv_h==8 and a
# kv_h==1 arch, while `k8v8` references a kernel set identical to `none`'s.
# The group size in the kernel name (gs_64) distinguishes the KV codec from the
# weight quantiser, which is gs_128/gs_32 on these snapshots.
_READS_PACKED = {"Mixed", "RotK"}

# Affine scale+bias sideband, bits per group. MEASURED off a real MixedTuple by
# `affine_sideband_is_thirty_two_bits_per_group` (crates/rmlx-kv-quant/src/
# quant_tests.rs): mx.quantize(mode="affine") emits scales and biases at the
# input dtype, and the KV stream reaching the store is bf16 because
# `cast_store_bf16` floors it there -- two bytes each, 32 bits per group.
_AFFINE_SIDEBAND_BITS = 32

# Store group sizes, from source: q8.rs Q8_GROUP_SIZE and turboquant.rs GROUP_SIZE.
Q8_GROUP_SIZE = 128
TURBO_GROUP_SIZE = 32


def _plane_bytes(codes_per_row: int, bits: int, n_tokens: int) -> int:
    """Bytes the dense code plane holds for `n_tokens` rows.

    Mirror of `code_plane::row_words` + `plane_bytes` (quant.rs): codes are
    packed LSB-first across a row's groups and the row is padded to a whole u32.
    """
    words = -(-(codes_per_row * bits) // 32)
    return words * 4 * n_tokens


def _affine_side_bytes(bits: int, group: int, elems: int) -> int:
    """Packed affine bytes for one axis: codes plus the per-group scale+bias."""
    if group <= 0:
        group = 64
    return int(elems * (bits + _AFFINE_SIDEBAND_BITS / group) / 8)


# quant.rs -- feeds_bf16_k/v_at_decode, the two arms that answer a constant.
#
# CAUTION: this predicate is a SEED-ALLOCATION gate, not a decode-read flag --
# "must exit_prefill materialise this buffer for SOME consumer?" Membership is
# therefore necessary but NOT sufficient for reads-mirror; _READS_PACKED
# overrides it.
_NO_BF16_K = {"IsoKOnly3", "IsoKOnly4", "RotorKOnly3", "RotorKOnly4",
              "Iso3Sym", "Iso4Sym", "Rotor3Sym", "Rotor4Sym"}
_NO_BF16_V = {"Iso3Sym", "Iso4Sym", "Rotor3Sym", "Rotor4Sym"}

# The two variants whose mirror is not a codec property: their bf16 K/V pair
# exists for a cross-layer-KV consumer and nothing else reads it, so the answer
# is the cache's own `shares_kv` on both axes. Every other variant answers a
# constant from the sets above.
_MIRROR_FOLLOWS_SHARES_KV = {"Mixed", "RotK"}


def feeds_bf16_k_at_decode(kind: str, shares_kv: bool) -> bool:
    """Mirror of `KvQuant::feeds_bf16_k_at_decode` (quant.rs)."""
    if kind in _NO_BF16_K:
        return False
    if kind in _MIRROR_FOLLOWS_SHARES_KV:
        return shares_kv
    return True


def feeds_bf16_v_at_decode(kind: str, shares_kv: bool) -> bool:
    """Mirror of `KvQuant::feeds_bf16_v_at_decode` (quant.rs)."""
    if kind in _NO_BF16_V:
        return False
    if kind in _MIRROR_FOLLOWS_SHARES_KV:
        return shares_kv
    return True


_ISO = {"Iso3", "Iso4", "Iso3Sym", "Iso4Sym", "IsoKOnly3", "IsoKOnly4"}
# The iso codecs whose resident form is the GPU ring (codes/scales/norms, no
# quaternion -- the rotation is the compile-time FIXED_QUAT). Their fused append
# seeds the ring from the prefill CPU blocks and then drops them
# (`drop_blocks_when_ring_live_iso_*`, kvcache/update.rs), so the ring is the
# sole resident copy from the first fused decode step. `Iso3` / `Iso4` have no
# ring path and would hold the CPU-block form, 6.07x larger (43.25 bits/value
# against the ring's 7.125 at head_dim 128 -- the same code plane, but the
# blocks carry f32 scales, a replicated f32 quaternion per group and an f32
# norm). Mirrors `SideStore::IsoRing` vs `SideStore::IsoBlocks`.
_ISO_RING = {"Iso3Sym", "Iso4Sym", "IsoKOnly3", "IsoKOnly4"}
_ROTOR = {"Rotor3", "Rotor4", "Rotor3Sym", "Rotor4Sym", "RotorKOnly3",
          "RotorKOnly4", "RotorK3Asym", "RotorK4Asym"}
_PLANAR = {"Planar", "Planar3", "PlanarK"}
# Which side actually carries the planar/iso/rotor encoding. Mirrors
# `KvQuant::side_stores`.
_K_FAMILY = {"PlanarK", "Iso3Sym", "Iso4Sym", "IsoKOnly3", "IsoKOnly4",
             "Rotor3Sym", "Rotor4Sym", "RotorKOnly3", "RotorKOnly4",
             "RotorK3Asym", "RotorK4Asym"}
_V_FAMILY = {"Planar", "Planar3", "Iso3", "Iso4", "Iso3Sym", "Iso4Sym",
             "Rotor3", "Rotor4", "Rotor3Sym", "Rotor4Sym"}

_MIXED_RE = re.compile(r"^mixed_k(\d+)g(\d+)_v(\d+)g(\d+)$")
_ROTK_RE = re.compile(r"^rot_k_v(\d+)g(\d+)$")
_ROTOR_ASYM_RE = re.compile(r"^rotor_k_([34])_asym_v(\d+)_g(\d+)$")

VALID_CODECS = (
    "none, k8v4, k8v8, planar, planar3, planar_k, k8vturbo3, k8vturbo3tcq, "
    "k8vturbo2tcq, tsym3, tsym4, k8vturbo2, iso3, iso4, iso3_sym, iso4_sym, "
    "k_iso3, k_iso4, rotor3, rotor4, rotor3_sym, rotor4_sym, k_rotor3, "
    "k_rotor4, rotor_k_3_asym_v<vb>_g<vg>, rotor_k_4_asym_v<vb>_g<vg>, "
    "rot_k_v<vb>g<vg>, mixed_k<kb>g<kg>_v<vb>g<vg>"
)


def parse_codec(s: str) -> Codec:
    """Mirror of `<KvQuant as FromStr>::from_str` (quant.rs)."""
    if s in _SIMPLE:
        kind, kb, vb = _SIMPLE[s]
        canonical = "none" if kind == "None" else s
        return Codec(canonical, kind, kb, vb)
    m = _MIXED_RE.match(s)
    if m:
        kb, kg, vb, vg = (int(x) for x in m.groups())
        return Codec(s, "Mixed", kb, vb, kg, vg)
    m = _ROTK_RE.match(s)
    if m:
        vb, vg = (int(x) for x in m.groups())
        return Codec(s, "RotK", 8, vb, 64, vg)
    m = _ROTOR_ASYM_RE.match(s)
    if m:
        kb, vb, _vg = (int(x) for x in m.groups())
        return Codec(s, f"RotorK{kb}Asym", kb, vb)
    raise SystemExit(f"unknown KvQuant '{s}' -- valid: {VALID_CODECS}")


def _side_bytes(c: Codec, bits: int, elems: int, n_tokens: int, head_dim: int,
                uses_family: bool, retains_seed: bool, group: int) -> int:
    """Per-side stored+seed bytes. Mirrors the `side_bytes` closure and
    `packed_side_bytes` in quant.rs; the family split mirrors `SideStore`."""
    if bits >= 16:
        return elems * BF16
    if uses_family and c.kind in _PLANAR:
        # PlanarQuant: 4 u32 code words + 2 u32 rotation words per 32-element
        # group, and one f32 scale PER PAIR (storage/quant_planar_v.rs, in the
        # `gpu_codes_buf.is_none()` init; quant_planar_k.rs is bit-identical).
        # 22.00 bits per value at every head_dim and at BOTH bit widths -- the
        # 3-bit pack is 10 vals/u32, ceil(32/10) = 4 words, the same word count
        # as 4-bit's 8 vals/u32. Reading it as codes plus one f32 per 32 values
        # would give 5.0 bits for the 4-bit members and 4.0 for planar3 --
        # understating a store that is ABOVE bf16's 16.0 by 4.4x and 5.5x, and
        # in the codec's favour.
        groups = elems // TURBO_GROUP_SIZE
        stored = groups * 4 * 4 + (elems // 2) * 4 + groups * 2 * 4
    elif uses_family and c.kind in _ISO:
        # Ring-backed: the row's dense code plane -- one code per value, `bits`
        # each -- plus one sideband scale per quaternion group and one sideband
        # norm per token. Block-backed: the host `Vec` form, same plane, but its
        # scale, norm and the 16B quaternion it replicates per group are all f32
        # and not the ring's dtype (`SideStore::IsoBlocks`).
        plane = _plane_bytes(head_dim, bits, n_tokens)
        if c.kind in _ISO_RING:
            stored = (plane + (elems // 4) * _RING_SIDEBAND_BYTES
                      + n_tokens * _RING_SIDEBAND_BYTES)
        else:
            stored = plane + (elems // 4) * (4 + 16) + n_tokens * 4
    elif uses_family and c.kind in _ROTOR:
        # group size 3: per-token ceil(head_dim/3), NOT elems/3. Three codes per
        # group -- the grade-1 components, the only ones stored.
        n_groups = -(-head_dim // 3)
        stored = (_plane_bytes(n_groups * 3, bits, n_tokens)
                  + n_groups * n_tokens * _RING_SIDEBAND_BYTES
                  + n_tokens * _RING_SIDEBAND_BYTES)
    else:
        codes = elems * bits // 8
        # Sideband cadence is per-store, not a single constant, and the Rust
        # `SideStore` arms are now per-store too -- Q8, Turbo and Affine{group},
        # each byte-exact against its own encoder
        # (`every_codec_byte_model_matches_the_store_it_writes`). The three
        # branches below are those three arms, and the two models agree:
        #   q8_0 K   -- Q8_GROUP_SIZE = 128 (q8.rs:33), one f32 per group
        #   TurboQuant V -- GROUP_SIZE = 32 (turboquant.rs:70), one f32 per group
        #   Mixed/RotK affine -- group from the codec name; the store is an
        #     mx.quantize 3-tuple, so the sideband is a scale AND a bias at the
        #     input dtype, 32 bits per group on the bf16 stream a cache holds
        #     (measured: `affine_sideband_is_thirty_two_bits_per_group`).
        # Those three are the WHOLE list this arm serves. Planar used to fall in
        # here unnamed and was modelled at 5.0 against a measured 22.0; it now
        # has its own branch above. An audit that enumerates the stores it
        # checked is only as good as the enumeration.
        # Footprint figures only: decode_read_bytes_per_layer never reaches here.
        if c.kind in _READS_PACKED:
            # The affine group is this axis's own, passed in by the caller. It
            # cannot be recovered from `bits` here: a codec whose two axes share
            # a width but not a group (mixed_k8g128_v8g32) would take the K
            # group for both.
            scales = int(elems / (group or 64)) * 2 * BF16
        elif bits == 8:
            scales = (elems // Q8_GROUP_SIZE) * 4
        else:
            scales = (elems // TURBO_GROUP_SIZE) * 4
        stored = codes + scales
    return stored + (elems * BF16 if retains_seed else 0)


# quant.rs -- KvQuant::decode_reads_packed_store. Transcribed ARM FOR ARM from
# the Rust match; do not derive it from the _NO_BF16_* sets, which happen to
# overlap it today and would silently misclassify the first codec that reads its
# store AND keeps both mirrors -- precisely the case the Rust predicate exists to
# express.
_DECODE_READS_PACKED_STORE = {
    # quantized-SDPA over the affine 3-tuples, appended per step
    "Mixed", "RotK",
    # K re-quantised into the packed store every decode step
    "IsoKOnly3", "IsoKOnly4", "RotorKOnly3", "RotorKOnly4",
    # flash decode straight off both packed rings
    "Iso3Sym", "Iso4Sym", "Rotor3Sym", "Rotor4Sym",
}


def decode_reads_packed_store(kind: str) -> bool:
    """Mirror of `KvQuant::decode_reads_packed_store` (quant.rs)."""
    return kind in _DECODE_READS_PACKED_STORE


def materialises_packed_store(kind: str) -> bool:
    """Mirror of `KvQuant::materialises_packed_store` (quant.rs).

    Rust: `decode_reads_packed_store() || !feeds_bf16_k || !feeds_bf16_v`. A
    codec that feeds both axes from the bf16 mirror and has no decode path over
    its packed store never gets one built -- `exit_prefill` skips the bulk
    encode, so its resident KV is the two mirrors and nothing else.

    This is a SECOND producer of a classification whose first producer is the
    Rust enum, and nothing gates them against each other. When a codec is added
    or reclassified, both move together or this roofline is off by a full store
    per layer, silently, in a direction that flatters the codec.
    """
    return (decode_reads_packed_store(kind)
            or not feeds_bf16_k_at_decode(kind, False)
            or not feeds_bf16_v_at_decode(kind, False))


def resident_bytes_per_layer(c: Codec, seq: int, head_dim: int,
                             kv_heads: int, shares_kv: bool) -> int:
    """Mirror of `KvQuant::estimated_resident_bytes_per_layer` (quant.rs).

    `shares_kv` is the stack's cross-layer-KV topology, the same flag the cache
    carries. It moves `Mixed` / `RotK` and nothing else, and it moves them by two
    whole bf16 mirrors -- 6.50 against 22.50 bits per value at group 64. Pricing
    them at a fixed `True` reports the one compressing family in the tree as a
    memory regression on every dense architecture.
    """
    elems = seq * head_dim * kv_heads
    if c.kind == "None":
        return elems * BF16 * 2
    if not materialises_packed_store(c.kind):
        # Both mirrors, no store -- byte-identical to `none` at this shape.
        return elems * BF16 * 2
    n_tokens = seq * kv_heads
    k = _side_bytes(c, c.k_bits, elems, n_tokens, head_dim,
                    c.kind in _K_FAMILY,
                    feeds_bf16_k_at_decode(c.kind, shares_kv), c.k_group)
    v = _side_bytes(c, c.v_bits, elems, n_tokens, head_dim,
                    c.kind in _V_FAMILY,
                    feeds_bf16_v_at_decode(c.kind, shares_kv), c.v_group)
    return k + v


def decode_read_bytes_per_layer(c: Codec, seq: int, head_dim: int,
                                kv_heads: int) -> int:
    """Bytes of KV a single decode step streams for one layer.

    This is NOT the resident figure.

    Three decode-read behaviours, established by GPU capture:

      * reads-packed via quantized_matmul -- Mixed / RotK (_READS_PACKED).
      * reads-packed via a fused flash kernel -- the *_sym and K-only families
        (the genuine feeds_bf16_* == false arms).
      * reads-mirror -- everything else, streaming the full bf16 warm-TTFT seed.

    Do NOT infer the bucket from feeds_bf16_* alone; see the caution above it.
    docs/PERF_BASELINE.md:1003 ("every mode reads bf16 K+V") generalises a
    three-cell table into a universal and is false for Mixed / RotK.

    KNOWN GAPS, both of which make this an optimistic lower bound on divergence:

      1. Gate state is not modelled. k8v4 / planar* / tsym* are classified
         reads-mirror only because --turbo-flash, --fused-qk and
         --planar-flash-decode all resolve OFF on this host. `--fused-qk on`
         flips K8V8 to reads-packed and this function will not notice.
      2. Shared-KV topology is not modelled. On those archs the producing layer
         reads packed while consumers attend over the surfaced bf16 prefix.

    The iso/rotor stored terms use the ring layout (codes/scales/norms), which
    is what a decode kernel actually streams. Only the four ring-backed iso
    codecs reach this arm at all: `Iso3` / `Iso4` keep both bf16 mirrors and take
    the reads-mirror branch above.
    """
    elems = seq * head_dim * kv_heads
    if c.kind == "None":
        return elems * BF16 * 2
    n_tokens = seq * kv_heads
    if c.kind in _READS_PACKED:
        return (_affine_side_bytes(c.k_bits, c.k_group, elems)
                + _affine_side_bytes(c.v_bits, c.v_group, elems))
    if feeds_bf16_k_at_decode(c.kind, False):
        k = elems * BF16
    else:
        k = _side_bytes(c, c.k_bits, elems, n_tokens, head_dim,
                        c.kind in _K_FAMILY, False, c.k_group)
    if feeds_bf16_v_at_decode(c.kind, False):
        v = elems * BF16
    else:
        v = _side_bytes(c, c.v_bits, elems, n_tokens, head_dim,
                        c.kind in _V_FAMILY, False, c.v_group)
    return k + v


# The width the boundary promotion floors both axes to. Mirror of
# `boundary_floor`'s FLOOR_BITS (kv_cache/mod.rs).
_BOUNDARY_FLOOR_BITS = 8


def boundary_floor(base: Codec, shares_kv: bool) -> Codec:
    """Mirror of `boundary_floor` (kv_cache/mod.rs).

    The promotion is a quality floor at 8 bits, not a codec switch. A base whose
    widths are parameters carries its own 8-bit form and is raised inside its
    family -- same store, same group geometry, same K rotation. A base whose
    width is baked into its variant has no such form and falls back to `k8v8`.

    A target that still mirrors both axes decodes at model dtype whatever its
    store holds, so the store buys no floor and is charged on top of the
    mirrors; `k8v8` is those same mirrors without the store. The final arm
    diverts such a target, read off the target's own predicates rather than a
    variant list.
    """
    if base.kind == "Mixed":
        target = Codec(
            f"mixed_k{max(base.k_bits, _BOUNDARY_FLOOR_BITS)}g{base.k_group}"
            f"_v{max(base.v_bits, _BOUNDARY_FLOOR_BITS)}g{base.v_group}",
            "Mixed",
            max(base.k_bits, _BOUNDARY_FLOOR_BITS),
            max(base.v_bits, _BOUNDARY_FLOOR_BITS),
            base.k_group,
            base.v_group,
        )
    elif base.kind == "RotK":
        # RotK's K is fixed at 8-bit/group-64 and is already at the floor; only
        # its V carries a width.
        v_bits = max(base.v_bits, _BOUNDARY_FLOOR_BITS)
        target = Codec(f"rot_k_v{v_bits}g{base.v_group}", "RotK", 8, v_bits,
                       64, base.v_group)
    else:
        target = parse_codec("k8v8")
    if (feeds_bf16_k_at_decode(target.kind, shares_kv)
            and feeds_bf16_v_at_decode(target.kind, shares_kv)):
        return parse_codec("k8v8")
    return target


def kv_quant_for_layer(idx: int, n_layers: int, base: Codec,
                       shares_kv: bool) -> Codec:
    """Mirror of `kv_quant_for_layer` (kv_cache/mod.rs). The first HEAD_N and
    last TAIL_N layers take `boundary_floor` for every base codec that quantizes
    at least one side. A base that keeps both sides at model dtype (16 bits —
    `none`) is exempt: the promotion recovers quantization loss, and there is
    none to recover."""
    if base.k_bits >= 16 and base.v_bits >= 16:
        return base
    is_tail = LAYER_ADAPTIVE_TAIL_N > 0 and idx >= n_layers - LAYER_ADAPTIVE_TAIL_N
    is_head = LAYER_ADAPTIVE_HEAD_N > 0 and idx < LAYER_ADAPTIVE_HEAD_N
    return boundary_floor(base, shares_kv) if (is_tail or is_head) else base


def next_pow2(n: int) -> int:
    """Mirror of `next_pow2_seq` (kvcache/update.rs:7982)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ── Model geometry ───────────────────────────────────────────────────────────

@dataclass
class KvLayer:
    idx: int
    head_dim: int
    kv_heads: int
    n_q_heads: int
    window: int | None      # None = global (full attention)
    owns_kv: bool           # allocates AND fills its own token-indexed cache
    kv_source: int          # index of the layer whose KV this layer reads


@dataclass
class ModelSpec:
    name: str
    arch: str
    n_layers: int
    layers: list[KvLayer]
    weight_bytes_step: int
    active_params_step: int
    quant_bits: int
    notes: list[str] = field(default_factory=list)


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def build_layers(cfg: dict) -> tuple[str, list[KvLayer], list[str]]:
    """Per-layer KV geometry. Keyed off config geometry only, mirroring how
    each arch builds its `KvLayerShape` vector."""
    tc = _text_cfg(cfg)
    arches = cfg.get("architectures") or []
    arch_name = arches[0] if arches else cfg.get("model_type", "unknown")
    n = int(tc["num_hidden_layers"])
    notes: list[str] = []
    hidden = int(tc.get("hidden_size", 0))
    n_q = int(tc.get("num_attention_heads", 0))
    kv_h = int(tc.get("num_key_value_heads", n_q))
    head_dim = int(tc.get("head_dim") or (hidden // n_q if n_q else 0))
    types = tc.get("layer_types") or []

    if "Gemma4" in arch_name or tc.get("model_type") == "gemma4_text":
        # gemma4/generate/mod.rs:346 -- sliding layers use head_dim +
        # num_key_value_heads; full layers use global_head_dim +
        # num_global_key_value_heads (config.rs:417, falls back to kv_heads).
        g_head_dim = int(tc.get("global_head_dim") or head_dim)
        g_kv_h = int(tc.get("num_global_key_value_heads") or kv_h)
        window = int(tc.get("sliding_window") or 0)
        shared = int(tc.get("num_kv_shared_layers") or 0)
        first_shared = n - shared
        layers: list[KvLayer] = []
        last_by_type: dict[str, int] = {}
        for i in range(n):
            lt = types[i] if i < len(types) else "full_attention"
            sliding = lt != "full_attention"
            if i < first_shared:
                last_by_type[lt] = i
            layers.append(KvLayer(
                idx=i,
                head_dim=head_dim if sliding else g_head_dim,
                kv_heads=kv_h if sliding else g_kv_h,
                n_q_heads=n_q,
                window=window if (sliding and window > 0) else None,
                owns_kv=i < first_shared,
                # loader.rs:360 build_previous_kvs
                kv_source=i if i < first_shared else last_by_type.get(lt, i),
            ))
        if shared:
            notes.append(
                f"num_kv_shared_layers={shared}: layers {first_shared}..{n - 1} "
                "own no KV (loader.rs:151) and re-read an earlier layer's cache"
            )
        return arch_name, layers, notes

    if "Qwen3_5Moe" in arch_name or "Qwen3_5" in arch_name or \
            tc.get("full_attention_interval"):
        # qwen3_5_moe/generate.rs:510 -- only every `full_attention_interval`-th
        # layer holds a token-indexed KV cache; the rest are GatedDeltaNet
        # linear-attention layers with a fixed-size recurrent state.
        interval = int(tc.get("full_attention_interval") or 4)
        layers = []
        for i in range(n):
            full = (i + 1) % interval == 0
            layers.append(KvLayer(i, head_dim, kv_h, n_q, None, full, i))
        n_full = sum(1 for x in layers if x.owns_kv)
        notes.append(
            f"full_attention_interval={interval}: {n_full}/{n} layers hold a "
            f"token-indexed KV cache; {n - n_full} GDN layers hold a "
            "fixed-size recurrent state the codec never touches"
        )
        return arch_name, layers, notes

    # Dense full-attention default (Qwen3 / Bonsai and friends). Honour
    # layer_types + sliding_window when the config declares them.
    window = int(tc.get("sliding_window") or 0) if tc.get("use_sliding_window") else 0
    layers = []
    for i in range(n):
        lt = types[i] if i < len(types) else "full_attention"
        sliding = lt != "full_attention"
        layers.append(KvLayer(
            i, head_dim, kv_h, n_q,
            window if (sliding and window > 0) else None, True, i,
        ))
    return arch_name, layers, notes


# ── Weight census from safetensors headers ───────────────────────────────────

_SKIP_PREFIXES = ("vision_tower", "audio_tower", "embed_vision", "embed_audio",
                  "multi_modal_projector", "vision_model", "audio_model")
# Looked up one row at a time at decode -- not streamed.
_LOOKUP_SUFFIXES = ("embed_tokens", "embed_tokens_per_layer")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def read_headers(snapshot: Path) -> list[tuple[str, str, list[int], int]]:
    out = []
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no *.safetensors under {snapshot.name}")
    for f in files:
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            off = v["data_offsets"]
            out.append((k, v["dtype"], list(v["shape"]), off[1] - off[0]))
    return out


def _bits_for(name: str, cfg: dict) -> int:
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    base = int(q.get("bits", 16))
    stem = name.rsplit(".", 1)[0]
    per = q.get(stem)
    if isinstance(per, dict) and "bits" in per:
        return int(per["bits"])
    return base


def weight_census(snapshot: Path, cfg: dict, spec_layers: list[KvLayer],
                  arch: str) -> tuple[int, int, list[str]]:
    """Bytes of weights streamed per decode step, plus the active parameter
    count used by the prefill FLOP model.

    Classification rules, in order:
      * multimodal towers          -> excluded (text decode never reads them)
      * embed_tokens*              -> excluded (one-row gather), EXCEPT when
                                      tie_word_embeddings makes embed_tokens
                                      the output projection, which IS streamed
      * rank-3 tensors             -> stacked MoE experts `[E, out, in]`;
                                      scaled by top_k/E (gather_qmm reads only
                                      the selected experts)
      * gemma4 k_proj/v_proj on a
        KV-shared tail layer       -> excluded (loader.rs:157 never loads them)
      * everything else            -> streamed in full
    """
    tc = _text_cfg(cfg)
    tied = bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)))
    top_k = int(tc.get("num_experts_per_tok") or tc.get("top_k_experts") or 0)
    owns_kv = {l.idx: l.owns_kv for l in spec_layers}
    is_gemma4 = "Gemma4" in arch

    total_bytes = 0
    total_params = 0
    notes: list[str] = []
    n_expert_tensors = 0
    for name, dtype, shape, nbytes in read_headers(snapshot):
        short = name.split(".", 1)[1] if name.startswith("language_model.") else name
        if any(short.startswith(p) or name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        stem = name.rsplit(".", 1)[0]
        leaf = stem.rsplit(".", 1)[-1]

        if leaf in _LOOKUP_SUFFIXES:
            if not (tied and leaf == "embed_tokens"):
                continue  # pure gather, or an untied input embedding

        if is_gemma4 and leaf in ("k_proj", "v_proj"):
            m = _LAYER_RE.search(name)
            if m and not owns_kv.get(int(m.group(1)), True):
                continue  # shared-KV tail layer: projection is never loaded

        frac = 1.0
        if len(shape) == 3 and top_k:
            n_experts = shape[0]
            frac = min(top_k, n_experts) / n_experts
            n_expert_tensors += 1

        bits = _bits_for(name, cfg)
        if dtype == "U32" and bits < 16:
            params = 1
            for d in shape[:-1]:
                params *= d
            params *= shape[-1] * (32 // bits)
        else:
            params = 1
            for d in shape:
                params *= d

        total_bytes += int(nbytes * frac)
        # Only `.weight` carries parameters; scales/biases are sidebands.
        if name.endswith(".weight"):
            total_params += int(params * frac)

    if n_expert_tensors:
        notes.append(
            f"MoE: {n_expert_tensors} stacked expert tensors scaled to "
            f"top_k={top_k} of the leading expert dim"
        )
    if tied:
        notes.append("tie_word_embeddings=true: embed_tokens counted once as "
                     "the streamed output projection")
    return total_bytes, total_params, notes


# ── KV totals per context ────────────────────────────────────────────────────

def stack_shares_kv(spec: ModelSpec) -> bool:
    """The stack's cross-layer-KV topology -- the flag the arch builder hands
    `KvCache::with_shares_kv`. True when any layer attends over another layer's
    cache, which is what `build_layers` records as `kv_source != idx`."""
    return any(l.kv_source != l.idx for l in spec.layers)


def kv_at_ctx(spec: ModelSpec, base: Codec, ctx: int,
              max_ctx: int | None) -> tuple[int, int, dict]:
    """Return (decode_read_bytes_per_step, resident_bytes, detail).

    Read bytes count EVERY layer's attention read, including a gemma4 shared-KV
    layer re-reading its source layer's cache -- that is a real DRAM stream per
    layer per step. Resident bytes count only layers that own a cache.
    """
    ceiling = max_ctx if max_ctx else ctx
    shares_kv = stack_shares_kv(spec)
    by_idx = {l.idx: l for l in spec.layers}
    read = 0
    resident = 0
    n_global = 0
    n_windowed = 0
    for l in spec.layers:
        src = by_idx[l.kv_source]
        if not src.owns_kv:
            continue
        seq_r = min(ctx, src.window) if src.window else ctx
        # An SWA layer always runs the bf16 rotating ring regardless of the
        # codec flag (kvcache/core.rs:346).
        codec_r = parse_codec("none") if src.window else \
            kv_quant_for_layer(src.idx, spec.n_layers, base, shares_kv)
        read += decode_read_bytes_per_layer(codec_r, seq_r, src.head_dim, src.kv_heads)

    for l in spec.layers:
        if not l.owns_kv:
            continue
        if l.window:
            n_windowed += 1
            # rotating.rs:7 -- ring is preallocated to exactly `max_size`
            cap = l.window
            codec = parse_codec("none")
        else:
            n_global += 1
            cap = min(next_pow2(ctx), ceiling)
            codec = kv_quant_for_layer(l.idx, spec.n_layers, base, shares_kv)
        resident += resident_bytes_per_layer(codec, cap, l.head_dim,
                                             l.kv_heads, shares_kv)
    return read, resident, {"n_global": n_global, "n_windowed": n_windowed}


# ── Prefill FLOP model + measured anchor ─────────────────────────────────────

def prefill_flops_per_token(spec: ModelSpec, ctx: int) -> float:
    """2*P for the projections, plus the attention score/context GEMMs.

    Averaged over a causal prefill of `ctx` tokens each token attends to
    ~ctx/2 keys, so per layer QK^T + AV cost 4*n_q_heads*head_dim*(ctx/2)
    = 2*n_q_heads*head_dim*ctx FLOPs per token. A windowed layer caps the
    attended span at its window.
    """
    f = 2.0 * spec.active_params_step
    for l in spec.layers:
        if not l.owns_kv:
            continue
        span = min(ctx, l.window) if l.window else ctx
        f += 2.0 * l.n_q_heads * l.head_dim * span
    return f


def prefill_anchor(db: Path, model_basename: str) -> dict | None:
    """Median measured prefill_tps for this model from runs.db, read-only.

    Reads `observations`, not `bests`: the anchor is the MEDIAN of a cell's
    measurements, and the view carries one champion row per cell, which would
    make the roofline anchor on a best-ever value instead of a typical one.
    That means this query has to carry the plausibility bound itself, so it is
    kept identical to METRICS_DB.md §4.1 for `prefill_tps` -- `Bounds::positive(1e5)`,
    i.e. `value > 0 AND value <= 1e5`. Do not "tighten" it here; change §4.1 and
    the registry, then mirror it.

    Also filters backend=rmlx and ts_utc >= PREFILL_ANCHOR_MIN_TS. Picks the
    cell with the most rows.
    """
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        rows = con.execute(
            "SELECT prompt_tokens, value, hardware_tag FROM observations "
            "WHERE metric='prefill_tps' AND backend='rmlx' AND model=? "
            "AND value>0 AND value<=1e5 AND ts_utc>=?",  # §4.1 Bounds::positive(1e5)
            (model_basename, PREFILL_ANCHOR_MIN_TS),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    cells: dict[int, list[float]] = {}
    tags: dict[int, str] = {}
    for pt, v, tag in rows:
        if pt:
            cells.setdefault(int(pt), []).append(float(v))
            tags[int(pt)] = tag
    best = None
    for pt, vs in cells.items():
        if len(vs) < PREFILL_ANCHOR_MIN_N:
            continue
        if max(vs) / min(vs) > PREFILL_ANCHOR_MAX_SPREAD:
            continue
        if best is None or len(vs) > len(cells[best]):
            best = pt
    if best is None:
        return None
    vs = cells[best]
    return {
        "prompt_tokens": best,
        "tps_median": statistics.median(vs),
        "tps_min": min(vs),
        "tps_max": max(vs),
        "n": len(vs),
        "hardware_tag": tags[best],
        "since": PREFILL_ANCHOR_MIN_TS,
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def analyse(snapshot: Path, codec_name: str, ctxs: list[int], args) -> dict:
    cfg = json.loads((snapshot / "config.json").read_text())
    arch, layers, geo_notes = build_layers(cfg)
    wbytes, wparams, w_notes = weight_census(snapshot, cfg, layers, arch)
    tc = _text_cfg(cfg)
    spec = ModelSpec(
        name=snapshot.name,
        arch=arch,
        n_layers=int(tc["num_hidden_layers"]),
        layers=layers,
        weight_bytes_step=wbytes,
        active_params_step=wparams,
        quant_bits=int((cfg.get("quantization") or {}).get("bits", 16)),
        notes=geo_notes + w_notes,
    )
    base = parse_codec(codec_name)
    bw = args.bandwidth_gbs * 1e9

    # Model name as recorded in runs.db: snapshot dir without the "<org>__" prefix.
    db_name = spec.name.split("__", 1)[-1]
    anchor = None
    achieved = args.prefill_tflops * 1e12 if args.prefill_tflops else PREFILL_ACHIEVED_FLOPS
    if achieved is None and not args.no_db:
        anchor = prefill_anchor(Path(args.runs_db), db_name)
        if anchor:
            achieved = anchor["tps_median"] * prefill_flops_per_token(
                spec, anchor["prompt_tokens"])

    rows = []
    for ctx in ctxs:
        kv_read, kv_res, det = kv_at_ctx(spec, base, ctx, args.max_ctx)
        bytes_step = wbytes + kv_read
        ceiling = bw / bytes_step
        if achieved:
            pf_tps = achieved / prefill_flops_per_token(spec, ctx)
            ttft = ctx / pf_tps * 1000.0
        else:
            pf_tps = None
            ttft = None
        rows.append({
            "model": spec.name,
            "arch": spec.arch,
            "kv_quant": base.name,
            "ctx": ctx,
            "weight_bytes_step": wbytes,
            "kv_bytes_step": kv_read,
            "bytes_step": bytes_step,
            "ceiling_tps": ceiling,
            "ms_per_token_floor": 1000.0 / ceiling,
            "kv_total_mb": kv_res / 1e6,
            "prefill_ceiling_tps": pf_tps,
            "ttft_floor_ms": ttft,
            "kv_frac": kv_read / bytes_step,
            "n_global_layers": det["n_global"],
            "n_windowed_layers": det["n_windowed"],
        })
    return {
        "host_bandwidth_gbs": args.bandwidth_gbs,
        "prefill_anchor": anchor,
        "prefill_achieved_tflops": achieved / 1e12 if achieved else None,
        "active_params_step": spec.active_params_step,
        "notes": spec.notes,
        "rows": rows,
    }


def print_table(res: dict) -> None:
    rows = res["rows"]
    r0 = rows[0]
    print(f"model      : {r0['model']}  ({r0['arch']})")
    print(f"kv_quant   : {r0['kv_quant']}")
    print(f"bandwidth  : {res['host_bandwidth_gbs']:.0f} GB/s "
          "(docs/PERF_BASELINE.md:101)")
    print(f"weights    : {r0['weight_bytes_step'] / 1e9:.3f} GB/step, "
          f"{res['active_params_step'] / 1e9:.2f}e9 active params")
    print(f"kv layers  : {r0['n_global_layers']} global, "
          f"{r0['n_windowed_layers']} windowed")
    a = res["prefill_anchor"]
    if a:
        print(f"prefill    : anchored on measured prefill_tps median "
              f"{a['tps_median']:.0f} @ {a['prompt_tokens']} tok "
              f"(n={a['n']}, {a['tps_min']:.0f}-{a['tps_max']:.0f}, "
              f"{a['hardware_tag']}, since {a['since']}) "
              f"-> {res['prefill_achieved_tflops']:.1f} TFLOP/s achieved")
    elif res["prefill_achieved_tflops"]:
        print(f"prefill    : {res['prefill_achieved_tflops']:.1f} TFLOP/s (--prefill-tflops)")
    else:
        print("prefill    : no trustworthy measured anchor -> null "
              "(pass --prefill-tflops to override)")
    for n in res["notes"]:
        print(f"note       : {n}")
    print()
    hdr = (f"{'ctx':>7} {'wt_GB':>7} {'kv_GB':>7} {'tot_GB':>7} "
           f"{'ceil_tps':>9} {'ms/tok':>7} {'kv_res_MB':>10} "
           f"{'pf_tps':>9} {'ttft_ms':>9} {'kv_frac':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pf = f"{r['prefill_ceiling_tps']:9.0f}" if r["prefill_ceiling_tps"] else f"{'null':>9}"
        tt = f"{r['ttft_floor_ms']:9.0f}" if r["ttft_floor_ms"] else f"{'null':>9}"
        print(f"{r['ctx']:>7} {r['weight_bytes_step'] / 1e9:7.3f} "
              f"{r['kv_bytes_step'] / 1e9:7.3f} {r['bytes_step'] / 1e9:7.3f} "
              f"{r['ceiling_tps']:9.1f} {r['ms_per_token_floor']:7.3f} "
              f"{r['kv_total_mb']:10.1f} {pf} {tt} {r['kv_frac']:8.3f}")


def emit_byte_model(lines: list[str]) -> int:
    """Re-emit a Rust byte-model manifest from this script's own arithmetic.

    Input is the manifest `emit_kv_byte_model_manifest` (rmlx-models) prints:
    the codec sweep, the shapes and the layer count all come from the Rust side,
    so this mode adds no second list of what to cover. Output carries the same
    key fields with this script's value in the last column, which is what
    `scripts/check_kv_byte_model_parity.sh` diffs.
    """
    rows = 0
    for raw in lines:
        f = raw.rstrip("\n").split("\t")
        if f[0] == "KVBYTES":
            name, flag, seq, head_dim, kv_heads = f[1], f[2], int(f[3]), int(f[4]), int(f[5])
            shares_kv = flag == "1"
            got = resident_bytes_per_layer(parse_codec(name), seq, head_dim,
                                           kv_heads, shares_kv)
            print(f"KVBYTES\t{name}\t{flag}\t{seq}\t{head_dim}\t{kv_heads}\t{got}")
            rows += 1
        elif f[0] == "KVFLOOR":
            name, flag, n_layers = f[1], f[2], int(f[3])
            shares_kv = flag == "1"
            base = parse_codec(name)
            # Every layer, not a sample: LAYER_ADAPTIVE_HEAD_N and
            # LAYER_ADAPTIVE_TAIL_N are copied by hand into this file, and only
            # a full vector puts both of them inside the diff.
            per_layer = ",".join(
                kv_quant_for_layer(i, n_layers, base, shares_kv).name
                for i in range(n_layers)
            )
            print(f"KVFLOOR\t{name}\t{flag}\t{n_layers}\t{per_layer}")
            rows += 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Theoretical decode/prefill ceiling for an rMLX model x KV codec.")
    p.add_argument("--byte-model", action="store_true",
                   help="read a Rust byte-model manifest on stdin and re-emit it "
                        "from this script's arithmetic; takes no model")
    p.add_argument("--model", help="model snapshot directory")
    p.add_argument("--kv-quant", help=f"KvQuant name; one of: {VALID_CODECS}")
    p.add_argument("--ctx", type=int, action="append",
                   help="context length in tokens (repeatable)")
    p.add_argument("--max-ctx", type=int, default=None,
                   help="ring ceiling for the resident figure (default: --ctx)")
    p.add_argument("--bandwidth-gbs", type=float, default=HOST_BW_BYTES_PER_S / 1e9)
    p.add_argument("--prefill-tflops", type=float, default=None,
                   help="override the achieved-GEMM anchor, in TFLOP/s")
    p.add_argument("--runs-db", default=".rmlx/metrics/runs.db",
                   help="metrics DB to read the prefill anchor from (read-only)")
    p.add_argument("--no-db", action="store_true", help="do not consult runs.db")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.byte_model:
        rows = emit_byte_model(sys.stdin.readlines())
        if rows == 0:
            raise SystemExit("--byte-model read no manifest rows on stdin")
        return

    for required in ("model", "kv_quant", "ctx"):
        if not getattr(args, required):
            raise SystemExit(f"--{required.replace('_', '-')} is required "
                             "unless --byte-model is passed")
    snapshot = Path(args.model).expanduser()
    if not (snapshot / "config.json").is_file():
        raise SystemExit(f"no config.json under {snapshot}")
    res = analyse(snapshot, args.kv_quant, sorted(set(args.ctx)), args)
    if args.json:
        json.dump(res, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print_table(res)


if __name__ == "__main__":
    main()
