# Owner(s): ["oncall: profiler"]
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch.profiler import (
    kineto_available,
    ProfilerActivity,
    profile,
    record_function,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TemporaryFileName,
    TestCase,
)


PROBE_NAME = "kineto_use_be_probe"


@unittest.skipIf(not kineto_available(), "Kineto is required")
class TestKinetoUseBE(TestCase):
    def _export_trace(self):
        with TemporaryFileName(mode="w+") as fname:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                with record_function(PROBE_NAME):
                    torch.add(torch.ones(4), torch.ones(4))
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                return json.load(f)["traceEvents"]

    def _probe_events(self, events):
        return [e for e in events if e.get("name") == PROBE_NAME]

    def test_begin_end_slices_when_enabled(self):
        with patch.dict(os.environ, {"KINETO_USE_BE": "1"}):
            events = self._export_trace()

        probe = self._probe_events(events)
        self.assertEqual(sorted(e["ph"] for e in probe), ["B", "E"])
        begin = next(e for e in probe if e["ph"] == "B")
        end = next(e for e in probe if e["ph"] == "E")
        self.assertGreater(end["ts"], begin["ts"])

    def test_complete_events_by_default(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("KINETO_USE_BE", None)
            events = self._export_trace()

        probe = self._probe_events(events)
        self.assertEqual([e["ph"] for e in probe], ["X"])
        self.assertIn("dur", probe[0])

    def test_zero_disables_begin_end_slices(self):
        with patch.dict(os.environ, {"KINETO_USE_BE": "0"}):
            events = self._export_trace()

        probe = self._probe_events(events)
        self.assertEqual([e["ph"] for e in probe], ["X"])
        self.assertIn("dur", probe[0])


GPU_CATEGORIES = ("kernel", "cuda_runtime")

# Kineto is initialized once per process, in the mode of the first profiling
# session. If a CPU-only session (as in TestKinetoUseBE) comes first, later
# sessions record no GPU activity, so each GPU case runs in a fresh process.
GPU_CHILD_SCRIPT = r"""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

x = torch.ones(1024, device="cuda")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    torch.add(x, x)
    torch.cuda.synchronize()
prof.export_chrome_trace(sys.argv[1])
"""


@unittest.skipIf(
    not (kineto_available() and torch.cuda.is_available()),
    "Kineto and a GPU are required",
)
class TestKinetoUseBEGPU(TestCase):
    def _export_trace(self, use_be):
        env = os.environ.copy()
        env.pop("KINETO_USE_BE", None)
        if use_be is not None:
            env["KINETO_USE_BE"] = use_be
        with TemporaryFileName(mode="w+") as fname:
            result = subprocess.run(
                [sys.executable, "-c", GPU_CHILD_SCRIPT, fname],
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]

        by_cat = {
            cat: [e for e in events if e.get("cat") == cat] for cat in GPU_CATEGORIES
        }
        for cat, cat_events in by_cat.items():
            self.assertTrue(cat_events, f"No {cat} events were recorded")
        return by_cat

    def _assert_complete_events(self, by_cat):
        for cat, cat_events in by_cat.items():
            self.assertEqual({e["ph"] for e in cat_events}, {"X"}, cat)
            self.assertTrue(all("dur" in e for e in cat_events), cat)

    def test_begin_end_slices_when_enabled(self):
        by_cat = self._export_trace("1")

        for cat, cat_events in by_cat.items():
            slices = {}
            for e in cat_events:
                slices.setdefault(e["args"]["correlation"], {})[e["ph"]] = e
            for correlation, pair in slices.items():
                self.assertEqual(sorted(pair), ["B", "E"], (cat, correlation))
                self.assertEqual(pair["B"]["name"], pair["E"]["name"])
                self.assertGreaterEqual(pair["E"]["ts"], pair["B"]["ts"])

    def test_complete_events_by_default(self):
        by_cat = self._export_trace(None)

        self._assert_complete_events(by_cat)

    def test_zero_disables_begin_end_slices(self):
        by_cat = self._export_trace("0")

        self._assert_complete_events(by_cat)


if __name__ == "__main__":
    run_tests()
