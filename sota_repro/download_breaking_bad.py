from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any


API = "https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:10.5683/SP3/LZNPKB"


def _metadata() -> list[dict[str, Any]]:
    with urllib.request.urlopen(API) as response:  # nosec B310 - fixed official HTTPS endpoint
        document = json.load(response)
    files = document["data"]["latestVersion"]["files"]
    return [{"id": item["dataFile"]["id"], "name": item["dataFile"]["filename"], "bytes": item["dataFile"].get("filesize", 0)} for item in files]


def _download(file_id: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"https://borealisdata.ca/api/access/datafile/{file_id}"
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:  # nosec B310 - fixed official HTTPS endpoint
        shutil.copyfileobj(response, output)
    os.replace(temporary, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download selected official Breaking Bad compressed archives")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subset", choices=["everyday", "artifact", "both"], default="both")
    parser.add_argument("--list", action="store_true", help="List Dataverse files without downloading")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    files = _metadata()
    wanted = {"data_split.tar.gz"}
    if args.subset in {"everyday", "both"}:
        wanted.add("everyday_compressed.zip")
    if args.subset in {"artifact", "both"}:
        wanted.add("artifact_compressed.zip")
    matching = [item for item in files if item["name"] in wanted]
    if args.list:
        for item in files:
            print(f"{item['name']}\t{item['bytes']}\t{item['id']}")
        return 0
    missing = wanted - {item["name"] for item in matching}
    if missing:
        raise RuntimeError(f"Official Dataverse metadata did not contain: {sorted(missing)}")
    output = Path(args.output_root)
    for item in matching:
        target = output / item["name"]
        if target.exists() and target.stat().st_size == item["bytes"]:
            print(f"[skip] {target}")
            continue
        print(f"[get] {item['name']} ({item['bytes']} bytes)")
        if not args.dry_run:
            _download(int(item["id"]), target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
