struct BetaRound;

impl RoundDrafter for BetaRound {
    fn propose(&mut self, ctx: &RoundCtx) -> u32 {
        let step = ctx.block;
        let base = step + 1;
        base * 2
    }
}
