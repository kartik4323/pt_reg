"""Path guards must keep working without Python 3.9's pathlib method."""
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from generative_assembly.storage import checked, is_relative_to
from generative_assembly.export import bundle


class PathCompatibilityTests(unittest.TestCase):
    def test_containment_is_not_a_string_prefix(self):
        for kind, root in ((PurePosixPath, '/study'), (PureWindowsPath, 'C:/study')):
            base=kind(root)
            self.assertTrue(is_relative_to(base,base))
            self.assertTrue(is_relative_to(base/'inputs'/'cloud.npz',base))
            self.assertFalse(is_relative_to(kind(root+'-other')/'cloud.npz',base))
        self.assertFalse(is_relative_to(PureWindowsPath('D:/study'),PureWindowsPath('C:/study')))

    def test_checked_without_new_pathlib_api_still_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'; root.mkdir()
            with patch.object(Path,'is_relative_to',create=True,side_effect=AttributeError('unavailable')):
                self.assertEqual(checked(root,'inputs/cloud.npz'),root.resolve()/'inputs'/'cloud.npz')
                self.assertEqual(checked(root,'.'),root.resolve())
                for escape in ('../outside.npz',str(root.parent/'study-other'/'cloud.npz')):
                    with self.assertRaisesRegex(ValueError,'escapes artifact root'):
                        checked(root,escape)

    def test_export_without_new_pathlib_api_still_rejects_nested_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            store=SimpleNamespace(root=root)
            with patch.object(Path,'is_relative_to',create=True,side_effect=AttributeError('unavailable')):
                with self.assertRaisesRegex(ValueError,'Export outside the run root'):
                    bundle(store,root/'nested-bundle','research')
            self.assertFalse((root/'nested-bundle').exists())


if __name__=='__main__': unittest.main()
