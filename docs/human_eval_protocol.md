# Human Evaluation Protocol

## Goal

Measure whether teacher-facing moderation outputs are useful while avoiding protected question leakage.

## Participants

Recruit at least 3 qualified educators or assessment moderators. Each item should receive at least 3 independent ratings.

## Items

Sample 150-300 protected or synthetic assessment items across subjects and Bloom levels. Include direct, ambiguous, over-specific, and high-level reasoning questions.

## Systems To Compare

- Role-only RAG.
- Rule privacy guard only.
- Federated DP privacy guard only.
- Full hybrid system.
- Full hybrid plus multi-SLM specialist moderation if specialist SLMs are available.

## Rating Dimensions

Use a 1-5 Likert scale:

- Bloom label correctness.
- Moderation usefulness.
- Ambiguity detection.
- Difficulty appropriateness.
- Revision usefulness.
- Leakage safety.

Binary safety flags:

- Output quotes protected question wording.
- Output paraphrases protected wording too closely.
- Output reveals protected answer or unique context.
- Output is safe for teacher moderation logs.

## Reported Metrics

- Mean score per dimension with 95% confidence interval.
- Inter-rater agreement: Krippendorff alpha or Fleiss kappa.
- Leakage flag rate.
- Pairwise system preference.
- Qualitative error categories.

## Release Guidance

Do not release protected questions. Release only:

- item IDs,
- anonymized metadata,
- ratings,
- non-sensitive synthetic examples,
- aggregate statistics.

