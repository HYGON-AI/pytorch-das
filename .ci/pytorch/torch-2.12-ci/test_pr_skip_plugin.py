import importlib.util
import sys
import types
import unittest
from pathlib import Path


CI_DIR = Path(__file__).resolve().parent
PLUGIN_PATH = CI_DIR / "pr_skip_plugin.py"
SKIP_FILE = CI_DIR / "pr-skipped-tests.txt"


class _FakeMark:
    @staticmethod
    def skip(reason):
        return ("skip", reason)


sys.modules.setdefault("pytest", types.SimpleNamespace(mark=_FakeMark()))
spec = importlib.util.spec_from_file_location("pr_skip_plugin", PLUGIN_PATH)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


class _FakeItem:
    def __init__(self, nodeid):
        self.nodeid = nodeid
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


class TestHcuPrSkipPlugin(unittest.TestCase):
    def test_marks_exactly_the_reviewed_cases(self):
        clauses = plugin.load_skip_clauses(SKIP_FILE)
        self.assertEqual(len(clauses), 7)

        reviewed_items = []
        for clause in clauses:
            class_name, test_name = clause.split(" and ")
            reviewed_items.append(
                _FakeItem("test/example.py::{}::{}".format(class_name, test_name))
            )

        retained_items = [
            _FakeItem(
                "test/test_nn.py::TestNN::"
                "test_batchnorm_2D_train_NCHW_vs_cpu_mixed_float16"
            ),
            _FakeItem(
                "test/test_reductions.py::TestReductionsCUDA::test_bincount_cuda"
            ),
        ]
        items = reviewed_items + retained_items

        old_value = plugin.os.environ.get(plugin.SKIP_FILE_ENV)
        plugin.os.environ[plugin.SKIP_FILE_ENV] = str(SKIP_FILE)
        try:
            plugin.pytest_collection_modifyitems(items)
        finally:
            if old_value is None:
                plugin.os.environ.pop(plugin.SKIP_FILE_ENV, None)
            else:
                plugin.os.environ[plugin.SKIP_FILE_ENV] = old_value

        self.assertTrue(all(len(item.markers) == 1 for item in reviewed_items))
        self.assertTrue(all(not item.markers for item in retained_items))

    def test_clause_requires_all_terms(self):
        clause = "TestGDS and test_gds_read_write_tensors"
        self.assertTrue(
            plugin.clause_matches_nodeid(
                clause,
                "test/test_cuda.py::TestGDS::test_gds_read_write_tensors",
            )
        )
        self.assertFalse(
            plugin.clause_matches_nodeid(
                clause,
                "test/test_cuda.py::OtherClass::test_gds_read_write_tensors",
            )
        )


if __name__ == "__main__":
    unittest.main()
