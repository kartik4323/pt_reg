"""Validate model interpreters and copy a frozen suite config into a new file."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def check_python(executable, modules, label):
    executable = str(Path(executable).expanduser().resolve())
    if not Path(executable).is_file():
        raise ValueError(label + " interpreter does not exist: " + executable)
    code = (
        "import importlib, json, sys\n"
        "for name in " + repr(modules) + ": importlib.import_module(name)\n"
        "import torch\n"
        "assert torch.cuda.is_available(), 'CUDA is unavailable in this interpreter'\n"
        "print(json.dumps({'python': sys.executable, 'torch': torch.__version__, "
        "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}))\n"
    )
    result = subprocess.run([executable, "-c", code], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(label + " preflight failed:\n" + (result.stderr or result.stdout)[-4500:])
    print(label + ": " + result.stdout.strip())
    return executable


def prepare(suite, output, image_python, mesh_python, worker_python=None):
    suite = Path(suite).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite config: " + str(output))
    try:
        output.relative_to(suite)
    except ValueError:
        pass
    else:
        raise ValueError("Write the new config outside the original suite root")
    plan = json.loads((suite / "plan.json").read_text(encoding="utf-8"))
    # Copy exact frozen settings, including all model revisions. No model locking
    # or package-default migration occurs here.
    config = plan["base_config"]
    image_python = check_python(image_python, ["torch", "diffusers", "numpy", "PIL"], "Images")
    mesh_python = check_python(mesh_python, ["torch", "diffusers", "huggingface_hub", "numpy", "PIL", "trimesh"], "InstantMesh")
    worker_python = check_python(worker_python or image_python, ["torch", "numpy", "scipy", "PIL", "trimesh"], "Suite worker / E6")
    config.setdefault("images", {})["python"] = image_python
    config.setdefault("reconstruction", {})["python"] = mesh_python
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print("New config created; original suite unchanged. Use a new GA_RUN_GROUP.")
    print("export GA_CONFIG=" + shlex.quote(str(output)))
    print("export GA_PYTHON=" + shlex.quote(worker_python))
    print("export GA_DATA=" + shlex.quote(plan["dataset"]))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--images-python", required=True)
    parser.add_argument("--reconstruction-python", required=True)
    parser.add_argument("--worker-python", help="Defaults to the image interpreter; must support torch and core geometry dependencies")
    args = parser.parse_args()
    prepare(args.from_suite, args.out, args.images_python, args.reconstruction_python, args.worker_python)


if __name__ == "__main__":
    main()
