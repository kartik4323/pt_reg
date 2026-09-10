# Implementation verification — 2026-09-10

These results verify software correctness and the first real ShapeNet geometry pilot on the local Windows CPU environment. They do not demonstrate learned assembly performance or A5000 memory/throughput.

## Completed

- **80 automated tests pass:** 20 data, 9 neural, 20 solver, 12 training lifecycle, 10 workflow, 4 Bash-runner, and 5 precision tests. Command: `python -m unittest discover -s tests -p 'test_reassembly*.py' -q`. Runner tests simulate CLI outputs to check stage ordering, arguments and early stopping; they do not measure GPU or learning performance.
- Five production preflight correctness checks pass. Maximum oracle transform error was `1.12e-15`; original-coordinate export error was below `2e-15` on deterministic fixtures, within the required `1e-5` tolerance.
- The full default configuration completed real forward/loss/backward/optimizer steps in all three stages on CPU, with batch 2, 1,024 points per fragment and chunked 32³ field evaluation. The recorded preflight took about 4 seconds on this host; reserved GPU memory is explicitly null.
- A separate small end-to-end fixture run exercised eight CLI operations: preparation, prepared-data preflight, three learned stages, evaluation, XYZ inference with scaffold plotting, and reporting. Ten synthetic meshes produced 30 complementary patterns. Only two optimizer updates per stage were run; the unfitted model returned explicit failures and the report correctly denied advancement.
- Exact-source GT-field and perturbed-field evaluation paths also executed successfully on the fixture data. Their results are labeled controls, not learned-model achievements.
- Source/pattern integrity hashes, same-run resume, completed-stage handoff, failed-assembly export, strict memory limits and matching experiment/report gates have regression coverage. Resumed training matches uninterrupted training exactly in deterministic CPU tests.
- `python -m compileall -q reassembly` and `git diff --check` passed.

Local runtime: Python 3.10, PyTorch 2.11.0+cpu, Trimesh 4.12.2, Manifold3D 3.5.3. Local verification artifacts were written to `D:/reassembly_v2_checks` after checking disk space. This avoids the local C: volume, whose free space was below the required 50 GiB. Managed check artifacts remain below 10 MiB.

## Real ShapeNet geometry pilot

An authorized replacement credential resolved the earlier authentication failure. The bottle archive was downloaded selectively: 89,458,230 bytes (about 85.3 MiB), with SHA-256 matching the provider's metadata. Credentials were not written to repository files or reports.

The first real run exposed a cleanup-order defect: duplicate faces in ShapeNet OBJ files can use distinct vertex indices. Cleanup now merges coincident vertices before duplicate-face removal, then orients the existing closed, connected surface without filling or remeshing it. Regression tests cover the double-sided OBJ case and orientation-only repair, alongside the existing cavity-preservation checks.

Repeating the same first 100 source objects with corrected cleanup produced:

| Result | Count |
|---|---:|
| Accepted source objects | 14 |
| Rejected: non-watertight | 85 |
| Rejected: disconnected | 1 |
| Accepted complementary fracture patterns | 102 |
| Two-piece / three-piece patterns | 61 / 41 |
| Easy / intermediate / hard patterns | 33 / 42 / 27 |

Preparation took about 44.6 seconds on this CPU host. Ten pattern attempts exhausted their retries: nine failed the easy contact-area criterion and one failed piece-volume balance. All accepted source/pattern content hashes and the dataset fingerprint verified. Archive, prepared data and local check artifacts total about 105 MiB, with about 98.8 GiB free on D:.

The accepted-source count is below the required 30, so no learning run was started. The source pool needs reassessment before the held-out experiment. These are results for the specified 100 candidates, not a yield estimate for the entire category.

Local artifacts: `D:/reassembly_v2/sources/02876657.zip`, `D:/reassembly_v2/prepared_corrected/manifest.json`, and `D:/reassembly_v2/geometry_pilot_summary.json`. The initial rejected run remains separately under `D:/reassembly_v2/prepared` for diagnosis; use the corrected manifest for inspecting accepted examples.

## Expanded source-pool reassessment

After the initial 100-source stop, a read-only audit of all 498 downloaded bottle meshes found 73 passing the same source validation. The explicit `configs/reassembly_v2_bottles498.yaml` configuration expands candidate inspection to 498, pins the audited source revision, and retains the 30-source gate, geometry rules and all model/training/resource limits. The default configuration still examines 100 candidates.

Full preparation with this configuration then completed on CPU:

| Result | Count |
|---|---:|
| Accepted sources with valid fracture patterns | 73 |
| Accepted patterns | 546 |
| Two-piece / three-piece patterns | 329 / 217 |
| Easy / intermediate / hard patterns | 183 / 219 / 144 |
| Train / validation / test / cut-holdout patterns | 418 / 48 / 69 / 11 |

Preparation took 235.7 seconds and reported `learning_ready: true`. All 73 source and 546 pattern hashes verified (73,474,130 payload bytes). Dataset fingerprint: `9e6a9677a1688b579ffe1ab205fddd0e4fa0815ea48332b9c9bfc6417d29b0d1`. Local manifest: `D:/reassembly_v2/bottles498/prepared/manifest.json`. Source split assignment still occurs before fracture generation; after geometry rejection, retained train/validation/test source counts are 56/6/11.

The server runner `scripts/run_reassembly_v2_pilot.sh` now provides preparation, CUDA preflight, fixed overfit, matched pilots, evaluation and report commands. Its 18 guide Bash blocks and embedded Python were syntax-checked. All four orchestration tests passed, including stopping before training after preparation/preflight failure and before held-out training after a failed overfit gate.

## Not yet measured

The reported Stage 1 CUDA stop exposed a numerical-control bug: training raised on an overflow before GradScaler could skip the optimizer update and lower its scale. The revised loop retries the same accumulated batch with restored Python/NumPy/Torch random state, and bounds persistent failures. Matching normalization, sigmoid/log probabilities and similarity calculations use float32 while learned MLPs retain AMP. Preflight uses the same optimizer-update path and accumulation and carries a new numerical-version signature.

Regression tests use actual CPU float16 matrix products and an enabled CPU GradScaler to produce overflow, verify that optimizer parameters/state remain unchanged on the failed attempt, and match a successful reference update after replay. Other tests cover persistent non-finite gradients, missing gradients, non-finite norms/losses, failure reports and weak-fracture matching gradients. This verifies the recovery mechanism; the old server log does not identify which parameter first became non-finite, so the exact CUDA trigger is unconfirmed. The deprecation warning is removed in the pinned Torch environment.

A separate Stage 1 numerical regression used the real prepared bottles at full geometry resolution (1,024 points, 256/128/64 groups, 128 channels, batch 2, accumulation 4), with CPU float16 autocast and CPU GradScaler test overrides. It completed **24 successful optimizer updates**, recovering from actual overflows at **updates 20 and 23** by reducing the scale **65,536 → 32,768 → 16,384** and replaying the accumulated batches. Checkpoint weights and Adam state are finite, and every initialized Adam step counter equals 24. Validation loss was 7.90200 at update 24. This took 874.2 seconds on CPU and does not measure CUDA throughput, GPU memory, or assembly success. Artifacts: `D:/reassembly_v2_checks/amp_regression_20260910T194306Z/`.

The current host has CPU-only PyTorch. The expanded dataset passes the 30-source geometry gate. The A5000 preflight, fixed 16-pattern learning check, bounded held-out pilots, and four-condition scaffold-benefit comparison remain pending GPU execution. No previous checkpoints were loaded, and no full training run was launched.

Follow [the runbook](REASSEMBLY_V2.md) for the measured experiment. Treat `advance_eligible: false` as the current research status; numerical fixture accuracy does not establish that the scaffold improves learned assembly.
