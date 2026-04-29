Privacy claim guidance

Use:
- "high measured resistance under a defined adversarial prompt set"
- "reduced reconstruction risk for protected exam uploads"
- "role-aware access control plus output screening"
- "ordinal structure is preserved better than exact class accuracy under domain shift"

Avoid:
- "perfect privacy"
- "complete security"
- "provably impossible to reconstruct"
- "fully generalizes across all educational datasets"

Safer paper framing:
- "Under the evaluated attack set, student-facing reconstruction attempts were blocked while benign study assistance and teacher moderation remained available."
- "Cross-domain evaluation shows substantial class-level degradation but comparatively stronger ordinal consistency."
- "The study evaluates cognitive robustness and privacy-constrained deployment under educational domain shift."
- "The LDL output is used as a distributional uncertainty signal; severe ordinal jumps trigger fallback rather than hard cognitive specialization."
- "Ordinal-threshold modeling improves the MoocRadar-to-Figshare ternary transfer selection, but ordinal benefits are direction-dependent."
- "Cue/content ablations suggest that Bloom verbs and question structure carry transferable signal, while topic vocabulary remains a major source of domain shift."

Claim hierarchy:
1. Primary claim: cognitive robustness under Figshare/MoocRadar domain shift.
2. Secondary claim: ordinal metrics explain failure modes better than exact accuracy alone.
3. Deployment claim: local CPU-only operation plus role-aware protected-resource policy is feasible under the evaluated constraints.
4. Privacy claim: measured resistance under a defined prompt taxonomy; InfoNCE retrieval leakage remains a negative result.
5. Method claim: ordinal-aware modeling is useful as a severe-error reduction strategy in selected transfer directions, not a universal replacement for all baselines.
