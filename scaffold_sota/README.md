# Independent scaffold-transfer experiment

This study tests whether a frozen predicted object scaffold improves existing SOTA assembly models. It is separate from the three-stage assembly project. The authoritative design is [PLAN.md](PLAN.md).

The first pilot uses Jigsaw, CCS and GARF on complete two/three-fragment bottles. A frozen v3 field is the main prior and v2 is a control. Native architecture adapters, independent training/evaluation, matched conditions, prior exports/caches, diagnostic probes and source-paired reports are implemented. See [RUNBOOK.md](RUNBOOK.md) for executable VM commands and [configs/pilot.json](configs/pilot.json) for the initial budget.

**Validation status:** CPU contract and orchestration tests are available. Native CUDA dependencies, A5000 memory feasibility, successful baseline learning and performance improvements still require execution on the VM with its real data. No trained recipient checkpoint or v3 artifact is bundled. Training completion is explicitly not learning acceptance.

| Architecture | Implemented scope |
|---|---|
| Jigsaw, CCS | Native bottle training and B0–B4; first pilot |
| GARF | Native feature pretraining, frozen extractor and flow assembly with B0–B4; first pilot |
| PMTR | Native pair matching, input-only graph assembly and B0–B4; second wave |
| PuzzleFusion++ | Native VQ-VAE pretraining and **denoiser-only** assembly with B0–B4; verifier/agglomeration are outside this variant |
| DiffAssemble | Native graph diffusion and B0–B4; native 1000-point layout |
| CMNet | Native equivariant matching and B0/B1/B2 pooled conditioning; spatial B3/B4 reject explicitly |
| GPAT | Native semantic target-surface adapter API; excluded from bottle CLI. PartNet reader, surface export and compatible environment remain a later track |

The [matching contracts](adapters/matching_BACKENDS.md) and [generative contracts](adapters/generative_BACKENDS.md) document native components, frames and explicit study adaptations. These are controlled architecture experiments, not claims of reproducing published benchmark scores.

## Ownership

All study-specific code, model patches, configuration and tests belong under `pt_reg/scaffold_sota/`. Training runs, data views, mutable source copies, caches, checkpoints and reports belong under this study's own external run root, proposed as `~/Kartik_23CS30026/scaffold_sota/` on the root filesystem. Verify the resolved mount before use. Use a separate environment namespace such as `scaffold-sota-<model>`; never install into an active three-stage or existing reproduction environment.

The implemented package includes:

```text
pt_reg/scaffold_sota/
  README.md
  PLAN.md
  configs/          study conditions and resource settings
  adapters/         native SOTA inputs, predictions and frame conversion
  priors/           frozen field readers and study-owned prior variants
  models/           shared equal-capacity conditioning branch
  evaluation/       common scoring, failures and paired comparisons
  __main__.py       independent launch, resume and diagnostic commands
  tests/            study-specific contract and implementation checks
```

Every future run records `experiment_id=scaffold_sota`, its own run/parent IDs, config hash, source revision, input manifest, prior hash and condition. Resume only a run from this study with matching lineage. A frozen imported checkpoint is an input, never a writable resume target.

## Reuse boundary

Reuse immutable datasets/splits, pinned SOTA source, pure geometry/sampling/metric utilities and completed scaffold artifacts with explicit provenance. Keep a pinned copy or revision/hash for shared code so ongoing work cannot silently change an active experiment. Add wrappers or versioned study-local copies when changed behavior is needed.

The scaffold provider contract is **input fragments → field/tokens and coordinate metadata**. Use only the encoder/field components needed to produce it. A checkpoint stored in an original `s3/` directory may supply frozen field weights, but this study never consumes the original Stage 3 poses, requires its matcher/solver to succeed, or resumes that training job. Continuous-field evaluation loads a private frozen copy; cached-token readers consume an immutable export.

This study does not modify `reassembly/`, `reassembly/repair/`, their configurations or diagnostic state, shared model registries, pinned upstream clones, or original runs/checkpoints/reports. It does not invoke the original repair orchestration or inherit its contact/assembly acceptance gates. Existing diagnostic results are cited as historical evidence, not merged into the study's new result tables.

If cross-fitting or broader datasets need new prior training, create a study-owned prior variant with separate weights and configuration here. Do not repair or extend the main three-stage implementation to satisfy this experiment. Recipient conditions still use frozen priors.

An unavailable v3 artifact delays only v3-dependent conditions. Native preparation/training and v2 controls can proceed independently. Keep each condition correctly labelled; do not silently substitute v2 for v3.

## Shared hardware, independent lifecycle

The available device is an RTX A5000 with 24,564 MiB. Schedule one GPU job at a time, waiting for other work to finish without interrupting it. Use measured memory preflight and the limits in the plan.

The supplied snapshot has about 161G free on `/` and 30G on `/data`. Use `/`, preserve at least 50 GiB free, and budget up to 75 GiB of additional storage for this study with a separate 15-GiB allowance for other work. Share only resource awareness, not output directories or lifecycle control. Results, claims and reports remain specific to this experiment.
