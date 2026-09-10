# CCS

Pinned official source is populated in `upstream/`. Use `setup_env.sh`,
`smoke.sh`, `train.sh`, and `test.sh`; outputs remain outside `upstream/`.

The reproduction profile deliberately replaces the upstream installation
snippet with an internally consistent legacy stack: Torch 1.10.2/CUDA 11.3,
PyTorch3D 0.7.2 built for Torch 1.10.2, Lightning 1.6.2/TorchMetrics 0.9.2,
NumPy 1.23.5, and MKL 2024.0.0. MKL is capped because later releases cause
`libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` with this Torch binary.

`setup` is safe to rerun. It updates an existing `sota-ccs` environment,
removes the mismatched pip PyTorch3D wheel from the upstream recipe when
present, rebuilds Chamfer and PointNet2 against the pinned Torch ABI, runs
`pip check`, and performs CPU import checks. `smoke` additionally runs the
compiled CUDA operators before training.

The Chamfer and PointNet2 builds need a CUDA 11 compiler, not only Conda's
CUDA runtime. Setup discovers `CUDA_HOME`, `nvcc` on `PATH`, cluster installs
under `/usr/local/cuda-11*`, and an environment-local toolkit. If none exists,
load the cluster's CUDA 11 module or install `cudatoolkit-dev=11.3.1` into the
CCS environment and rerun setup. Cleanup removes generated build artifacts
directly and does not invoke `setup.py clean`, which itself incorrectly
requires `CUDA_HOME` in these upstream extensions.
