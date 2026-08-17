#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("junit_test_stats.py")
SPEC = importlib.util.spec_from_file_location("junit_test_stats", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class JUnitTestStatsTest(unittest.TestCase):
    def test_aggregates_nested_suites_and_testcase_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir)
            report_dir = result_root / "pr-1-of-2" / "test-reports" / "python"
            report_dir.mkdir(parents=True)
            (report_dir / "results.xml").write_text(
                """<?xml version="1.0"?>
<testsuites tests="5">
  <testsuite name="one">
    <testcase name="pass" time="1.25" />
    <testcase name="fail" time="2"><failure /></testcase>
  </testsuite>
  <testsuite name="two">
    <testcase name="error" time="0.5"><error /></testcase>
    <testcase name="skip"><skipped /></testcase>
    <testcase name="namespaced"><x:skipped xmlns:x="urn:test" /></testcase>
  </testsuite>
</testsuites>
""",
                encoding="utf-8",
            )

            stats = MODULE.collect_junit_stats(result_root)

            self.assertEqual(stats["xml_files"], 1)
            self.assertEqual(stats["total"], 5)
            self.assertEqual(stats["passed"], 1)
            self.assertEqual(stats["failures"], 1)
            self.assertEqual(stats["errors"], 1)
            self.assertEqual(stats["skipped"], 2)
            self.assertAlmostEqual(stats["duration_seconds"], 3.75)
            self.assertAlmostEqual(stats["success_rate"], 100.0 / 3.0)

    def test_counts_invalid_xml_without_discarding_valid_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir)
            report_dir = result_root / "pr-1-of-2" / "test-reports"
            report_dir.mkdir(parents=True)
            (report_dir / "valid.xml").write_text(
                '<testsuite><testcase name="pass" /></testsuite>',
                encoding="utf-8",
            )
            (report_dir / "invalid.xml").write_text(
                "<testsuite>", encoding="utf-8"
            )

            stats = MODULE.collect_junit_stats(result_root)

            self.assertEqual(stats["xml_files"], 2)
            self.assertEqual(stats["invalid_xml_files"], 1)
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["passed"], 1)


if __name__ == "__main__":
    unittest.main()
