"""Selective ShapeNet bottle acquisition; never fetch the entire dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


BOTTLE_CATEGORY = "02876657"
SHAPENET_REPOSITORY = "ShapeNet/ShapeNetCore"


class _CredentialSafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(newurl).netloc != urlsplit(request.full_url).netloc:
            # Hugging Face may redirect to a signed CDN URL. Never forward its
            # account bearer token to that different host.
            redirected.remove_header("Authorization")
        return redirected


def acquire_bottles(output: Path, cfg: dict, resource_guard=None, dry_run: bool = False) -> dict:
    """Check access and exact bytes before streaming the single category archive.

    Access tokens are read through huggingface_hub's normal credential lookup and
    are never returned, persisted in reports, or printed. Existing archives are
    accepted by ``prepare_dataset`` without calling this network interface.
    """
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

    data = cfg.get("data", {})
    revision = data.get("source_revision", "main")
    url = hf_hub_url(SHAPENET_REPOSITORY, f"{BOTTLE_CATEGORY}.zip", repo_type="dataset", revision=revision)
    report = {"schema_version": 2, "repository": SHAPENET_REPOSITORY,
              "category": BOTTLE_CATEGORY, "revision": revision, "url": url}
    token = get_token()
    try:
        metadata = get_hf_file_metadata(url, token=token)
    except Exception as exc:
        # Avoid including URLs/headers from HTTP exception strings containing credentials.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        report.update(status="access_unavailable", http_status=status,
                      reason="ShapeNet access requires an authorized Hugging Face account and accepted dataset terms."
                      if status in (401, 403) else f"Metadata lookup failed ({type(exc).__name__}).")
        return report
    size = metadata.size
    if size is None or size <= 0:
        return dict(report, status="size_unavailable", reason="Refusing a download without a verified archive size.")
    report.update(archive_bytes=int(size), commit=metadata.commit_hash, etag=metadata.etag)
    if resource_guard is not None:
        resource_guard.check(additional_bytes=int(size))
    max_bytes = int(float(cfg.get("resources", {}).get("max_download_gib", 4)) * 1024**3)
    if size > max_bytes:
        return dict(report, status="size_limit", reason=f"Category archive exceeds configured {max_bytes} byte limit.")
    output = Path(output).resolve()
    target = output / f"{BOTTLE_CATEGORY}.zip"
    report["archive"] = str(target)
    if dry_run:
        return dict(report, status="available")
    output.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size:
            return dict(report, status="existing_size_mismatch", reason="Existing archive size differs; no file overwritten.")
        import zipfile
        if not zipfile.is_zipfile(target):
            return dict(report, status="invalid_existing_archive")
        return dict(report, status="already_present")
    partial = target.with_suffix(".zip.part")
    if partial.exists():
        return dict(report, status="partial_exists", reason="Review or remove the explicit partial archive before retrying.")
    pinned_url = hf_hub_url(SHAPENET_REPOSITORY, f"{BOTTLE_CATEGORY}.zip", repo_type="dataset",
                            revision=metadata.commit_hash or revision)
    request = Request(pinned_url, headers={"Authorization": f"Bearer {token}"} if token else {})
    digest = hashlib.sha256()
    transferred = 0
    try:
        with build_opener(_CredentialSafeRedirect()).open(request, timeout=60) as response, partial.open("xb") as stream:
            while True:
                chunk = response.read(4 * 1024**2)
                if not chunk:
                    break
                if transferred + len(chunk) > size:
                    raise ValueError("Server delivered more than the verified archive size")
                if resource_guard is not None:
                    resource_guard.check(additional_bytes=len(chunk))
                stream.write(chunk)
                digest.update(chunk)
                transferred += len(chunk)
        if transferred != size:
            raise ValueError("Downloaded archive does not match verified byte count")
        import zipfile
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC validation failed")
        partial.replace(target)
    except (HTTPError, OSError, ValueError) as exc:
        return dict(report, status="download_failed", received_bytes=transferred,
                    reason=f"Download failed ({type(exc).__name__}); partial archive retained for inspection.")
    report.update(status="downloaded", sha256=digest.hexdigest())
    (output / "acquisition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
