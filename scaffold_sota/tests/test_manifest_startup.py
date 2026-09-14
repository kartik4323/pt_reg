"""Startup discovery and integrity checks work before model environments exist."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scaffold_sota.manifests import (
    check_data, find_manifests, inspect_manifest, manifest_fingerprint, verify_manifest,
)


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    source, pattern = root / "source.npz", root / "pattern.npz"
    source.write_bytes(b"immutable-source-content")
    pattern.write_bytes(b"immutable-pattern-content")
    document = {"schema_version": 2, "sdf_convention": "negative_inside", "seed": 42,
                "config": {"data": {"preparation": "test"}},
                "sources": [{"source_id": "s1", "path": source.name,
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
                "patterns": [{"source_id": "s1", "pattern_id": "p1", "path": pattern.name,
                              "sha256": hashlib.sha256(pattern.read_bytes()).hexdigest(),
                              "pieces": 2, "split": "train", "cut_family": "random", "band": "easy"}]}
    path = root / "manifest.json"
    save(path, document)
    return path, document


def save(path, document):
    document["fingerprint"] = manifest_fingerprint(document)
    path.write_text(json.dumps(document), encoding="utf-8")


class ManifestStartupTests(unittest.TestCase):
    def test_preserves_shared_fingerprint_and_cli_result(self):
        from reassembly.prepare import manifest_fingerprint as shared_fingerprint
        with tempfile.TemporaryDirectory() as temporary:
            path, document = fixture(Path(temporary))
            self.assertEqual(manifest_fingerprint(document), shared_fingerprint(document))
            result = check_data(path)
            self.assertEqual(set(result), {"fingerprint", "verification", "patterns", "sources"})
            self.assertEqual(result["patterns"], 1)
            self.assertEqual(result["verification"]["verified_bytes"], 49)
            self.assertEqual(result["verification"]["content_split_separation"], "verified")

    def test_verifies_without_site_packages_in_separate_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = fixture(Path(temporary))
            code = ("import json,sys; from scaffold_sota.manifests import check_data,find_manifests; "
                    "print(json.dumps(check_data(sys.argv[1]))); "
                    "assert not any(m in sys.modules for m in ('torch','numpy','trimesh','reassembly.prepare'))")
            result = subprocess.run([sys.executable, "-S", "-c", code, str(path)],
                                    cwd=str(Path(__file__).resolve().parents[2]),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["verification"]["status"], "verified")

    def test_missing_placeholder_is_actionable_before_optional_imports(self):
        with self.assertRaisesRegex(ValueError, "Replace the example.*find-data"):
            check_data("/absolute/path/to/bottle/manifest.json")

    def test_asset_hashing_uses_bounded_reads(self):
        from scaffold_sota.manifests import _file_sha256
        reads = []
        class BoundedStream(io.BytesIO):
            def read(self, size=-1):
                reads.append(size)
                if not 0 < size <= 4 * 1024 ** 2:
                    raise AssertionError("Asset verification requested an unbounded read")
                return super().read(size)
        content = b"immutable asset bytes"
        with patch("scaffold_sota.manifests.Path.open", return_value=BoundedStream(content)):
            self.assertEqual(_file_sha256(Path("unused.npz")), hashlib.sha256(content).hexdigest())
        self.assertEqual(len(reads), 2)

    def test_tampered_assets_and_fingerprint_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, document = fixture(root)
            (root / "pattern.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_manifest(path)
            path, document = fixture(root)
            document["seed"] += 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                verify_manifest(path)
            self.assertFalse(inspect_manifest(path)["fingerprint_matches"])

    def test_duplicate_identifiers_paths_and_missing_assets_fail(self):
        mutations = [
            (lambda d: d["sources"].append(dict(d["sources"][0])), "duplicate asset path"),
            (lambda d: d["patterns"][0].update(source_id="unknown"), "unknown source"),
            (lambda d: d["patterns"][0].update(path="missing.npz"), "Missing prepared asset"),
            (lambda d: d["patterns"][0].update(path="../outside.npz"), "outside the prepared"),
            (lambda d: d["patterns"][0].update(path="C:\\outside.npz"), "must be relative"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for mutate, message in mutations:
                with self.subTest(message=message):
                    path, document = fixture(Path(temporary))
                    mutate(document)
                    save(path, document)
                    with self.assertRaisesRegex(ValueError, message):
                        verify_manifest(path)

    def test_known_splits_complete_piece_sets_and_cut_family_are_required(self):
        mutations = [({"pieces": 4}, "unsupported fragment"), ({"pieces": 2.0}, "unsupported fragment"),
                     ({"split": "validation"}, "unknown data split"),
                     ({"cut_family": "heldout_radial"}, "Held-out cut family"),
                     ({"cut_family": None}, "cut_family is required")]
        with tempfile.TemporaryDirectory() as temporary:
            for changes, message in mutations:
                with self.subTest(changes=changes):
                    path, document = fixture(Path(temporary))
                    document["patterns"][0].update(changes)
                    save(path, document)
                    with self.assertRaisesRegex(ValueError, message):
                        check_data(path)

    def test_source_identity_and_duplicate_content_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for separate_id in (False, True):
                path, document = fixture(root)
                source = dict(document["sources"][0], source_id="s2", path="source2.npz")
                if separate_id:
                    (root / source["path"]).write_bytes((root / "source.npz").read_bytes())
                    document["sources"].append(source)
                record = dict(document["patterns"][0], path="pattern2.npz", pattern_id="p2", split="test",
                              source_id="s2" if separate_id else "s1")
                (root / record["path"]).write_bytes((root / "pattern.npz").read_bytes())
                document["patterns"].append(record)
                save(path, document)
                with self.assertRaisesRegex(ValueError, "leaks across"):
                    check_data(path)
            document["patterns"][0]["split"] = "test"
            document["patterns"][1].update(split="cut_holdout", cut_family="heldout_radial")
            save(path, document)
            self.assertEqual(check_data(path)["patterns"], 2)

    def test_discovery_is_metadata_only_and_reports_missing_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = fixture(root / "first")
            second, _ = fixture(root / "second")
            (second.parent / "source.npz").unlink()
            with patch("scaffold_sota.manifests._file_sha256", side_effect=AssertionError("Discovery hashed assets")):
                result = find_manifests([root])
            self.assertEqual(len(result["candidates"]), 2)
            by_path = {r["path"]: r for r in result["candidates"]}
            self.assertTrue(by_path[str(first.resolve())]["assets_present"])
            self.assertFalse(by_path[str(second.resolve())]["assets_present"])
            self.assertEqual(by_path[str(second.resolve())]["missing_asset_count"], 1)
            self.assertFalse(result["truncated"])
            self.assertTrue(all(not r["asset_hashes_verified"] for r in result["candidates"]))

    def test_discovery_skips_environment_hidden_cache_and_upstream_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chosen, _ = fixture(root / "prepared")
            for name in (".hidden", ".git", "envs", "venv", "node_modules", "site-packages", "upstream", "caches"):
                fixture(root / name)
            report = find_manifests(root)
            self.assertEqual([c["path"] for c in report["candidates"]], [str(chosen.resolve())])
            self.assertEqual(report["skipped_directories"], 8)

    def test_discovery_does_not_follow_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "outside")
            (root / "search").mkdir()
            try:
                (root / "search" / "linked").symlink_to(root / "outside", target_is_directory=True)
            except OSError:
                self.skipTest("Current account cannot create symlinks")
            self.assertEqual(find_manifests(root / "search")["candidates"], [])

    def test_discovery_reports_depth_file_limits_and_malformed_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "nested" / "deeper")
            self.assertTrue(find_manifests(root, max_depth=0)["truncated"])
            self.assertTrue(find_manifests(root, max_files=1)["truncated"])
            (root / "bad_manifest.json").write_text("[]", encoding="utf-8")
            report = find_manifests(root)
            self.assertEqual(len(report["candidates"]), 1)
            self.assertEqual(len(report["errors"]), 1)
            self.assertIn("expected a JSON object", report["errors"][0]["error"])
            self.assertTrue(find_manifests(root / "missing")["truncated"])

    def test_directory_only_searches_have_an_entry_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(20):
                (root / str(index)).mkdir()
            report = find_manifests(root, max_files=3)
            self.assertTrue(report["truncated"])
            self.assertLessEqual(report["scanned_entries"], 6)

    def test_unrelated_manifest_formats_do_not_block_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "prepared")
            (root / "package_manifest.json").write_text(json.dumps({"name": "other-project"}), encoding="utf-8")
            (root / "old_manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            report = find_manifests(root)
            self.assertEqual(len(report["candidates"]), 1)
            self.assertEqual(report["ignored_manifests"], 2)
            self.assertEqual(report["errors"], [])

    def test_default_roots_include_home_and_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "home")
            fixture(root / "external")
            with patch("scaffold_sota.manifests.Path.home", return_value=root / "home"), \
                    patch.dict(os.environ, {"REASSEMBLY_ROOT": str(root / "external")}):
                report = find_manifests()
            self.assertEqual(len(report["search_roots"]), 2)
            self.assertEqual(len(report["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
