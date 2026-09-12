"""Resource monitoring must tolerate live atomic writes without hiding I/O errors."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnostics.reassembly_v2.runtime import Limits
from reassembly import resources


class LiveResourceTests(unittest.TestCase):
    def test_progress_rename_between_enumeration_and_stat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress, temporary = root / 'progress.json', root / 'progress.json.tmp'
            progress.write_bytes(b'old')
            temporary.write_bytes(b'new progress')
            names = [temporary.name, progress.name]

            def walk(*args, **kwargs):
                # os.walk observed the temporary file, then the worker published it.
                if temporary.exists():
                    os.replace(temporary, progress)
                yield str(root), [], names

            with patch.object(resources.os, 'walk', side_effect=walk):
                self.assertEqual(resources.tree_bytes(root), len(b'new progress'))
                # Exercise both scans used by the real supervisor's Limits.check.
                limit = Limits(root, root, None)
                with patch.object(resources.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 100 * resources.GIB})()):
                    limit.check()
            self.assertEqual(progress.read_bytes(), b'new progress')

    def test_removed_root_returns_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(resources.tree_bytes(Path(directory) / 'missing'), 0)

    def test_permission_and_io_errors_are_not_suppressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / 'protected'
            protected.write_bytes(b'data')
            original = Path.lstat
            for error in (PermissionError('denied'), OSError('device failure')):
                def fail(path):
                    if path == protected:
                        raise error
                    return original(path)
                with self.subTest(error=type(error).__name__), patch.object(Path, 'lstat', fail):
                    with self.assertRaises(type(error)):
                        resources.tree_bytes(root)

    def test_directory_scan_permission_error_is_not_suppressed(self):
        with tempfile.TemporaryDirectory() as directory:
            def walk(*args, onerror, **kwargs):
                onerror(PermissionError('denied directory'))
                return iter(())
            with patch.object(resources.os, 'walk', side_effect=walk):
                with self.assertRaises(PermissionError):
                    resources.tree_bytes(Path(directory))

    def test_storage_and_free_space_limits_still_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').write_bytes(b'large enough')
            with self.assertRaisesRegex(RuntimeError, 'Managed storage cap exceeded'):
                resources.ResourceGuard([root], cap_gib=1 / resources.GIB, min_free_gib=0).check()
            with patch.object(resources.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1})()):
                with self.assertRaisesRegex(RuntimeError, 'Insufficient free space'):
                    resources.ResourceGuard([root], min_free_gib=50).check()


if __name__ == '__main__':
    unittest.main()
