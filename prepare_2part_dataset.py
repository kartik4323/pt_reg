#!/usr/bin/env python3
"""
Filter the dataset down to 2-piece dev cases and preserve the full
schema_version=1 wrapper that the pipeline's inventory() check requires.
"""
import json
from pathlib import Path

DATASET_DIR = Path("/home/kpandey/generative_assembly/bottles498-inputs-v1")
in_path     = DATASET_DIR / "dataset.json"
out_path    = DATASET_DIR / "dataset_2parts.json"
MAX_CASES   = 10

data  = json.loads(in_path.read_text())

# Validate the source schema
if data.get("schema_version") != 1:
    print(f"WARNING: schema_version is {data.get('schema_version')}, expected 1")

cases = data.get("cases", [])

# Filter: exactly 2 pieces, dev split only
dev_2part = [c for c in cases if c.get("pieces") == 2 and c.get("split") == "dev"]

if not dev_2part:
    # Fallback: any 2-piece case across all splits
    dev_2part = [c for c in cases if c.get("pieces") == 2]

if not dev_2part:
    print("ERROR: No 2-piece cases found at all!")
    raise SystemExit(1)

selected = dev_2part[:MAX_CASES]

# ── Preserve ALL top-level keys from original (schema_version, provenance, etc.)
# Only replace the 'cases' list with our filtered subset
out_data = {k: v for k, v in data.items() if k != "cases"}
out_data["cases"] = selected

out_path.write_text(json.dumps(out_data, indent=2))

print(f"✅ Total 2-piece cases available : {len(dev_2part)}")
print(f"✅ Selected for study            : {len(selected)}")
print(f"✅ schema_version preserved      : {out_data['schema_version']}")
print(f"✅ Saved to                      : {out_path}")
print()
for c in selected:
    npz = DATASET_DIR / c["path"]
    print(f"  {'✅' if npz.exists() else '❌'}  {c['id']}  ({c['difficulty_band']})  →  {c['path']}")
