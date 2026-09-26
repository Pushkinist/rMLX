//! Dense gate-up-down MLP and MoE stub.

use rmlx_core::error::Result;
use rmlx_mlx::{gelu_tanh, multiply, silu, Array, Device};

use super::linear::Linear;

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

/// Activation function for MLP layers.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — MLP activation functions; adding a variant requires updating Mlp::forward and all arch MLP construction sites"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// GELU with tanh approximation — used by Gemma4.
    GeluTanh,
    /// SiLU (swish) — used by Llama, Qwen.
    Silu,
    // Relu2 (max(x,0)^2) is NOT a variant here. BitNet inlines relu2 in
    // bitnet/model.rs::BitNetMlp::forward (it also has an ffn_sub_norm step
    // between relu2 and down_proj that the shared Mlp cannot express).
    // Add Relu2 here only when a second architecture needs it via shared Mlp.
}

// ---------------------------------------------------------------------------
// Mlp
// ---------------------------------------------------------------------------

/// Dense gate-up-down FFN.
///
/// Computes: down_proj(activation(gate_proj(x)) * up_proj(x))
///
/// Both Gemma4 (GeluTanh) and Llama/Qwen (SiLU) fit this shape.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed layer struct — fields are the complete gate-up-down FFN contract; adding a field requires updating all arch MLP construction sites"
)]
#[allow(
    missing_debug_implementations,
    reason = "Linear does not implement Debug; Mlp cannot derive it until Linear does"
)]
/// Dense gate-up-down FFN (`down_proj(act(gate_proj(x)) * up_proj(x))`).
pub struct Mlp {
    /// First linear in the gate branch.
    pub gate_proj: Linear,
    /// First linear in the up branch.
    pub up_proj: Linear,
    /// Output projection.
    pub down_proj: Linear,
    /// Activation function applied to the gate output.
    pub activation: Activation,
}

impl Mlp {
    /// Run the gate-up-down FFN forward pass.
    pub fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = self.gate_proj.forward(x, device)?;
        let gate = match self.activation {
            Activation::GeluTanh => gelu_tanh(&gate, device)?,
            Activation::Silu => silu(&gate, device)?,
        };
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}

// ---------------------------------------------------------------------------
// MoeBlock (stub)
// ---------------------------------------------------------------------------

/// Sparse mixture-of-experts FFN — an unimplemented stub.
///
/// The MoE architectures (e.g. `Qwen3_5MoeForConditionalGeneration`) route
/// experts in their own modules and never use this type.
#[allow(
    clippy::exhaustive_structs,
    reason = "unimplemented stub — placeholder with no fields"
)]
#[allow(missing_debug_implementations)]
pub struct MoeBlock {
    _placeholder: (),
}

impl MoeBlock {
    #[allow(
        clippy::unimplemented,
        reason = "MoeBlock::forward is an unimplemented stub; the MoE archs use their \
                  own paths and never call this."
    )]
    /// Unimplemented stub. See struct-level doc.
    pub fn forward(&self, _x: &Array, _device: Device) -> Result<Array> {
        unimplemented!("MoeBlock::forward is not implemented; the MoE archs use their own paths")
    }
}
