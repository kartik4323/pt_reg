"""Export saved repair metrics and geometry, excluding checkpoints/source assets."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def export(experiment, output):
    source, output = Path(experiment).resolve(), Path(output).resolve()
    if not source.is_dir(): raise FileNotFoundError(source)
    if output.exists(): raise FileExistsError(output)
    files = []
    for phase in ('overfit', 'contacts', 'preflight'):
        for name in ('config.resolved.json', 'training_report.json', 'history.jsonl', 'run.json', 'preflight.json'):
            path = source/phase/name
            if path.is_file(): files.append(path)
    for relative in ('gate', 'contact_validation'):
        for path in (source/relative).rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.npz'): files.append(path)
    for name in ('experiment.json',):
        if (source/name).is_file(): files.append(source/name)
    files = sorted(set(files))
    if not files: raise ValueError('No diagnostic artifacts found')
    if any(source not in path.resolve().parents for path in files):
        raise ValueError('Diagnostic inputs must not link outside the experiment')
    hashes = {str(path.relative_to(source)).replace('\\', '/'): digest(path) for path in files}
    manifest = json.dumps(dict(kind='repair_diagnostic_bundle', checkpoint_assets_included=False,
        prepared_assets_included=False, files=hashes), indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open('xb') as stream:
            created = True
            with tarfile.open(fileobj=stream, mode='w:gz') as archive:
                for path in files:
                    archive.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
                record = tarfile.TarInfo('bundle_manifest.json'); record.size = len(manifest)
                archive.addfile(record, io.BytesIO(manifest))
        if any(digest(path) != hashes[path.relative_to(source).as_posix()] for path in files):
            raise ValueError('Input artifacts changed during export; wait for the evaluation to finish')
    except BaseException:
        # Only remove the newly created file; pre-existing outputs are rejected above.
        if created and output.exists(): output.unlink()
        raise
    return dict(output=str(output), files=len(files), sha256=digest(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.experiment, args.output), indent=2))


if __name__ == '__main__': main()
