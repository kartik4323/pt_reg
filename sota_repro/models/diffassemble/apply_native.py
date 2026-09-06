"""Apply the guarded 3D-only import patch to a disposable source copy."""
import argparse
import hashlib
import json
from pathlib import Path


RELATIVE = 'puzzle_diff/model/backbones/__init__.py'


def apply_native(source, report):
    source, report = Path(source).resolve(), Path(report)
    if source.name == 'upstream' or (source / '.git').exists():
        raise ValueError('Refusing to patch an upstream/Git checkout; use a disposable native source copy')
    target = source / RELATIVE
    target.resolve().relative_to(source)
    patch = Path(__file__).parent / 'patches/3d-backbone-imports.patch'
    lines = patch.read_text(encoding='utf-8').splitlines(keepends=True)
    if lines[:2] != [f'--- a/{RELATIVE}\n', f'+++ b/{RELATIVE}\n']:
        raise ValueError('Unexpected patch target')
    # This is one full-file replacement, not a general patch interpreter.
    before = ''.join(line[1:] for line in lines[3:] if line.startswith('-'))
    after = ''.join(line[1:] for line in lines[3:] if line.startswith('+'))
    current = target.read_text(encoding='utf-8')
    if not before or current not in (before, after):
        raise RuntimeError(f'Patch preimage mismatch: {target}; refusing to overwrite local changes')
    record = dict(target=RELATIVE, already_applied=current == after,
                  before_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest())
    if current != after:
        target.write_text(after, encoding='utf-8')
    record['after_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('Applied recorded 3D-only backbone import patch to', source, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    apply_native(args.source, args.report)
