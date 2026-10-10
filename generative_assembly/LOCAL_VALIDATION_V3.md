# Local v3 validation — 2026-10-10

Outcome: implementation checks passed. **No real completion-model or InstantMesh
inference was run locally, and no final-model acceptance claim is made.**

Environment: Windows, Python 3.10, Torch 2.11.0+cpu; CUDA unavailable. VM setup uses
separate pinned Python 3.11 image/matte environments and the existing InstantMesh env.

## Checks actually run

Post-pilot model-lock repair: the full suite passed **59 tests** in 78.507 seconds.
New coverage rejects querying the community dataset as a model, preserves requested
author-pipeline revisions, and verifies local custom-code loading independently of
the model-weight revision. A live CPU-only Hub check resolved and downloaded
`sudo-ai/zero123plus-pipeline/pipeline.py` at commit
`983e66d28a3637ddd8e3e2fd8165cdff32230872`, with SHA256
`d33babc5d138b832294da9e18496c52d0ea7006b47464d4e5b1bd60cd7e37acd`.
This verifies code availability, not successful GPU reconstruction.

- `python -m unittest discover -s generative_assembly/tests -v`: **58 passed**
  (23 existing + 35 new), 76.955 seconds in the final run.
- `python -m compileall -q generative_assembly`: passed.
- Git Bash `bash -n run_imagination_study.sh` and
  `bash -n setup_imagination_v3.sh`: passed.
- `python -m generative_assembly.cpu_diagnostics --out outputs_imagination_v3_delivery --execute-notebook`:
  CPU E0/E1/E2/E4 smoke integration passed; **10 notebook code cells executed,
  zero error outputs**. Model workers were explicit smoke substitutes.
- `python -m generative_assembly.study_cycle collect --root outputs_imagination_v3_delivery --out outputs_imagination_v3_local/delivery-results.zip`:
  collector passed; local ZIP size 931,691 bytes, excluding checkpoint weights.
- `git diff --check`: passed (Windows line-ending notices only).

Tests cover bounded surfels, holes/determinism/provenance, camera projection/up-axis
rotation, dense public-input sampling, alpha validity/pale pixels/backgrounds,
RGBA normalization and independent transforms, SD15/SDXL/Qwen mocked interfaces,
pinned upstream preprocessing/custom revisions, GPU visibility/admission/locks,
budget refusal/accounting, failed-worker diagnostics and stale-attempt isolation,
continuous similarity/holdouts, missing-surface metrics, mesh-surface diversity,
shortfalls/original-coordinate exports, GT-free abstention/anchor preservation,
source balancing, frozen-cohort enforcement, failed dependencies, source-bootstrap
promotion gates, oracle exclusion, stress inputs and notebook execution.

## CPU diagnostic observations

The same public box-fragment fixture rendered with 32 assembly-sample splats had
0.62% canvas coverage; 1,024 bounded observed surfels had 6.10%. The shared comparison
camera was held fixed. The denser render represents observed geometry, not a
completed shape; this comparison is neither bottle performance nor benchmark evidence.

Unchanged smoke images had zero matte foreground growth, selected zero generated
completion priors, recorded explicit shortfalls, and triggered geometry-only
`B2__deploy` abstention. Public inference was separately tested with reference access
patched to raise an error. A mocked 10-dev/20-test source-disjoint cycle froze its
configuration, retained failure-inclusive denominators and blocked subsequent refreeze.

Generated local evidence is in ignored `outputs_imagination_v3_delivery/`:
`validation.json`, `render_comparison.png`, the executed notebook, and smoke artifacts.
These files are not included in the code commit. Nonfatal notebook warnings concerned
missing legacy cell IDs and Windows ZMQ event-loop compatibility; execution completed.

## Still requiring VM evidence

Real locked ONNX segmentation, diffusion model compatibility/quality/runtime/memory,
InstantMesh preprocessing and extraction on the pinned checkout, the actual oracle
6.2 ratio's PCA/component geometry, useful three-prior coverage, and measured assembly
gains. The local pinned ONNX file/rembg environment was unavailable, so retrospective
gallery segmentation was explicitly skipped, without a heuristic substitute.

Run the phase commands in `IMAGINATION_V3.md`. The historical one-source study cannot
satisfy the 10-development/20-untouched-test-source requirement. If data, model
capabilities or the 24-hour budget are insufficient, the pipeline reports preliminary
evidence and does not export an approved configuration.
