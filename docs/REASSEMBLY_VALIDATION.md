# Implementation verification — 2026-09-10

These results verify software correctness and the first real ShapeNet geometry pilot on the local Windows CPU environment. They do not demonstrate learned assembly performance or A5000 memory/throughput.

## Completed

- **68 automated tests pass:** 20 data, 9 neural, 20 solver, 10 training lifecycle, and 9 workflow tests. Command: `python -m unittest discover -s tests -p 'test_reassembly*.py' -q`.
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

## Not yet measured

The current host has CPU-only PyTorch. The A5000 preflight, fixed 16-pattern learning check, bounded held-out pilots, and four-condition scaffold-benefit comparison remain pending source-pool reassessment and a passing 30-source geometry gate. No previous checkpoints were loaded, and no full training run was launched.

Follow [the runbook](REASSEMBLY_V2.md) for the measured experiment. Treat `advance_eligible: false` as the current research status; numerical fixture accuracy does not establish that the scaffold improves learned assembly.
