"""Handle the legacy Torch executable-stack header on newer glibc hosts only."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def torch_import():
    return subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = torch_import()
    record = dict(action='none', initial_returncode=result.returncode, initial_output=result.stdout)
    if result.returncode and 'libtorch_cpu.so: cannot enable executable stack' in result.stdout:
        if version('torch').split('+')[0] != '1.12.1':
            raise RuntimeError('Executable-stack repair is audited only for Torch 1.12.1')
        library = Path(sysconfig.get_path('purelib')) / 'torch/lib/libtorch_cpu.so'
        library.resolve().relative_to(Path(sys.prefix).resolve())
        patcher = Path(sys.prefix) / 'bin/patchelf'
        record.update(action='clear-execstack', library=str(library), original_sha256=digest(library))
        backup = library.with_name(library.name + '.before-noexecstack')
        if not backup.exists():
            shutil.copy2(library, backup)
        record['backup'] = str(backup)
        # Conda files can be hardlinks shared with caches/other environments.
        # Patch a private copy, then atomically replace this environment's link.
        fd, temporary = tempfile.mkstemp(prefix='libtorch_cpu-noexec-', suffix='.so', dir=library.parent)
        os.close(fd)
        try:
            shutil.copy2(library, temporary)
            subprocess.run([str(patcher), '--clear-execstack', temporary], check=True)
            os.replace(temporary, library)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        record['patched_sha256'] = digest(library)
        print('Cleared legacy executable-stack flag in this environment only; original preserved at', backup, flush=True)
        result = torch_import()
    record.update(passed=result.returncode == 0, final_output=result.stdout)
    args.report.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f'Torch binary import failed; see {args.report}')


if __name__ == '__main__':
    main()
