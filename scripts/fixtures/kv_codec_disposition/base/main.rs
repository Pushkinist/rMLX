// Synthetic stand-in for crates/rmlx-cli/src/main.rs, read only by
// scripts/check_kv_codec_disposition_fixtures.sh. It is not compiled and is not
// part of any crate: the gate reads six help constants and the `--kv-quant` /
// `--kv-bits` / `--kv-preset` argument declarations out of the source, and each
// fixture makes one edit to this file, to KV_QUANT.md beside it, or to
// manifest.raw.
//
// The codec names are invented (`fixbase`, `fixinert`, `fixinert2`, `fixlive`)
// so a fixture can never collide with a real codec's stem. `fixinert2` extends
// `fixinert` on purpose: it keeps the gate's EXACT-stem word fencing under test,
// because a fixture that removes one must not look like it removed the other.

const KV_QUANT_HELP: &str = "\
KV cache quantization codec. Default \"auto\" = unquantised fixbase. \
No codec in the tree holds less resident KV than the baseline — see --help.";

const KV_QUANT_LONG_HELP: &str = "\
KV cache quantization codec. Default \"auto\".

  fixbase (what auto resolves to)
      The unquantised baseline, and the smallest resident KV.

  INERT — accepted, but does nothing:
      fixinert, fixinert2.

  Runs its codec — and measures LARGER than the baseline:
      fixlive.";

const KV_BITS_LONG_HELP: &str = "\
Bit-width alias for KV cache quantization. Names no codec of its own; see
--kv-quant for what each one does. Run `rmlx info --list-cache-types` for the
computed per-codec figures.";

const CACHE_TYPE_K_LONG_HELP: &str = "\
Per-side KV cache codec for the K tensor. Names cache-type tags, not KvQuant
names; see --kv-quant for what each codec does.";

const CACHE_TYPE_V_LONG_HELP: &str = "\
Per-side KV cache codec for the V tensor. Names cache-type tags, not KvQuant
names; see --kv-quant for what each codec does.";

const KV_PRESET_LONG_HELP: &str = "\
Named KV-cache preset. Each name resolves to a codec.

  INERT — every preset target except the baseline:
      fixslow resolves to fixinert.

Presets:
  fixbaseline -- the unquantised baseline
  fixslow     -- resolves to fixinert; see INERT above";

fn info(list_cache_types: bool) {
    if list_cache_types {
        print_cache_type_table();
        print_kv_quant_residency_table();
    }
}

enum Cmd {
    Serve {
        #[arg(long, help = KV_QUANT_HELP, long_help = KV_QUANT_LONG_HELP)]
        kv_quant: Option<String>,
        /// Integer bit-width KV quantization alias. See long-help.
        #[arg(
            long,
            value_name = "BITS",
            long_help = KV_BITS_LONG_HELP,
        )]
        kv_bits: Option<f32>,
        /// Named KV-cache preset. See long-help.
        #[arg(
            long,
            value_name = "NAME",
            long_help = KV_PRESET_LONG_HELP,
        )]
        kv_preset: Option<KvPresetArg>,
    },
    Chat {
        #[arg(
            long,
            default_value = "auto",
            help = KV_QUANT_HELP,
            long_help = KV_QUANT_LONG_HELP
        )]
        kv_quant: String,
        /// Integer bit-width KV quantization alias. See long-help.
        #[arg(
            long,
            value_name = "N",
            long_help = KV_BITS_LONG_HELP,
        )]
        kv_bits: Option<f32>,
        /// Named KV-cache preset. See long-help.
        #[arg(
            long,
            value_name = "NAME",
            long_help = KV_PRESET_LONG_HELP,
        )]
        kv_preset: Option<KvPresetArg>,
    },
}
