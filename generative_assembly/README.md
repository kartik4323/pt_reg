# Generative fragment assembly experiments

Independent implementation of E0–E7 from the fragment → image completion → complete shape → rigid assembly study. Start with [RUNBOOK.md](RUNBOOK.md). The research design is in [PROTOCOL.md](research/PROTOCOL.md).

For data reuse, start with [DATA_PREPARATION.md](DATA_PREPARATION.md). `import-v2` reuses the existing reassembly/scaffold bottle fragments in a separate directory. `demo` generates box fragments solely for software checks; the research commands do not silently substitute those boxes for real data.

The package implements a classical contact-proposal baseline, CPU rendering, pretrained image workers, an InstantMesh adapter, oracle controls, input-only template selection and gating, a small PointNet assembly learner, corruption experiments, reference-separated evaluation and reusable artifact bundles. It does not modify the existing repair/scaffold-transfer pipelines.

The CPU smoke configuration copies input images and substitutes boxes for reconstructed shapes. It tests software, not scientific feasibility. Pipeline export refuses smoke artifacts. Real generation and GPU memory/performance require the documented GPU environment and checkpoint downloads.

```bash
python -m generative_assembly --help
python -m unittest discover -s generative_assembly/tests -v
```

The [artifact contract](ARTIFACTS.md) describes the saved inputs, cameras, hypotheses, poses, pseudo-labels and checkpoints. `pipeline_recipe.json`, a complete source snapshot and `SHA256SUMS.json` support later pipeline integration without reconstructing experiment history.
