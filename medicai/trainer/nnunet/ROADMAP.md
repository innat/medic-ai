# nnU-Net Roadmap

This branch is building a Keras 3 nnU-Net pipeline for MedicAI.
The current goal is parity with official nnU-Net v2 behavior, not just a similarly named API.

## Current Status

- Done: starter modules for fingerprinting, planning, preprocessing, training, inference, and CV.
- Done: typed config dataclasses and richer preprocessing case properties.
- Done: safer split normalization and dataset case-id handling.
- In progress: align planner, sampler, augmentation, trainer, inference, and model orchestration with nnU-Net v2.

## Phase 1: Data Contracts

- Stabilize `DatasetFingerprint`, `nnUNetPlan`, and per-case preprocessing metadata.
- Persist case properties needed for inversion and patch sampling.
- Ensure case IDs are subject-level, not modality-file-level.

## Phase 2: Planner Parity

- Tighten target spacing heuristics.
- Match patch-size and batch-size planning more closely to nnU-Net v2.
- Add proper cascade planning for `3d_lowres -> 3d_cascade_fullres`.
- Support configuration ranking and empirical model selection.

## Phase 3: Preprocessing + Sampling

- Add richer preprocessing outputs and inversion utilities.
- Build a real patch sampler using foreground location caches.
- Respect configuration-specific preprocessing for `2d`, `3d_fullres`, and `3d_lowres`.

## Phase 4: Training Parity

- Implement the missing nnU-Net augmentation recipe.
- Align deep supervision targets and weighting.
- Add resume, validation export, and fold artifact conventions.
- Add distributed execution support consistent with MedicAI runtime patterns.

## Phase 5: Inference + Postprocessing

- Add preprocessing inversion during prediction.
- Add fold ensembling.
- Add cascade refinement inference.
- Learn and apply postprocessing rules from validation predictions.

## Phase 6: Verification

- Golden tests for fingerprint/plan outputs.
- Roundtrip preprocessing/inference tests.
- End-to-end smoke tests for `2d`, `3d_fullres`, and cascade paths.
- Compare behavior on a small public reference dataset.
