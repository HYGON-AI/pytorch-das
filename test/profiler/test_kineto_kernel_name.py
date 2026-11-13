# Owner(s): ["oncall: profiler"]
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.profiler import kineto_available
from torch.testing._internal.common_utils import (
    IS_LINUX,
    run_tests,
    TEST_WITH_ROCM,
    TestCase,
)


# hipModuleLaunchKernel callbacks used to store only the raw hipFunction_t and
# resolve the name with hipKernelNameRef() while the trace was serialized.
# Short-lived kernels (small FFTs, for example) unload their module right after
# the launch, so that lookup dereferenced a stale handle and dumped core.
# 0d7ee9e resolves the name in the launch callback instead.
#
# The guard below is preloaded into the profiled process and reports any
# hipKernelNameRef() call made once the workload is done, i.e. while the
# profiler is stopping and serializing the trace. The child arms it inside the
# profile block, after the last kernel is synchronized. Calls made by the
# launch callbacks themselves pass through.
GUARD_SOURCE = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef const char* (*name_ref_fn)(void*);

static name_ref_fn real_name_ref(void) {
  static name_ref_fn fn;
  static int resolved;
  if (!resolved) {
    resolved = 1;
    // RTLD_NEXT cannot see the versioned symbol, so resolve against the HIP
    // runtime directly. It is always loaded by the time a kernel is launched.
    void* handle = dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD);
    if (handle == NULL) {
      handle = dlopen("libamdhip64.so", RTLD_NOW);
    }
    if (handle != NULL) {
      fn = (name_ref_fn)dlsym(handle, "hipKernelNameRef");
    }
  }
  return fn;
}

const char* hipKernelNameRef(void* function) {
  const char* phase = getenv("KINETO_HIPNAMEREF_GUARD_PHASE");
  if (phase != NULL && strcmp(phase, "export") == 0) {
    fputs(
        "KINETO_STALE_HIPKERNELNAMEREF: hipKernelNameRef() was called while "
        "the trace was being serialized, after the launch callback had "
        "returned. The hipFunction_t may belong to an already-unloaded module. "
        "Capture the kernel name in the launch callback (0d7ee9e).\n",
        stderr);
    fflush(stderr);
    _exit(78);
  }
  name_ref_fn real = real_name_ref();
  return real != NULL ? real(function) : NULL;
}
"""

GUARD_EXIT_CODE = 78

CHILD_SCRIPT = r"""
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

trace_path = sys.argv[1]
unload_plans = sys.argv[2] == "1"
torch.cuda.init()
x = torch.randn(16, 16, device="cuda", dtype=torch.complex64)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    torch.fft.fft(x)
    torch.fft.ifft(x)
    torch.cuda.synchronize()
    if unload_plans:
        # Destroy the cached rocFFT plans so their code objects are unloaded
        # before the trace is serialized.
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.cuda.synchronize()
    # Launch callbacks have all fired by now; anything looked up from here on
    # is serialized after the fact.
    os.environ["KINETO_HIPNAMEREF_GUARD_PHASE"] = "export"

prof.export_chrome_trace(trace_path)
"""

def _uses_rocprofiler_sdk():
    with open("/proc/self/maps") as f:
        return "librocprofiler-sdk" in f.read()


USES_ROCPROFILER_SDK = IS_LINUX and _uses_rocprofiler_sdk()


@unittest.skipIf(
    not (
        IS_LINUX
        and TEST_WITH_ROCM
        and kineto_available()
        and torch.cuda.is_available()
    ),
    "Requires Linux, ROCm, Kineto and a GPU",
)
class TestKinetoKernelName(TestCase):
    def _export_guarded_trace(self, unload_plans):
        cc = os.environ.get("CC", "cc").split()[0]
        if shutil.which(cc) is None:
            self.skipTest("A C compiler is required to build the guard")

        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "hipnameref_guard.c")
            guard = os.path.join(tmpdir, "hipnameref_guard.so")
            trace = os.path.join(tmpdir, "trace.json")
            with open(source, "w") as f:
                f.write(GUARD_SOURCE)
            subprocess.run(
                [cc, "-shared", "-fPIC", "-O2", "-o", guard, source, "-ldl"],
                check=True,
                capture_output=True,
            )

            env = os.environ.copy()
            preload = env.get("LD_PRELOAD")
            env["LD_PRELOAD"] = f"{guard}:{preload}" if preload else guard
            result = subprocess.run(
                [sys.executable, "-c", CHILD_SCRIPT, trace, str(int(unload_plans))],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if (
                result.returncode == GUARD_EXIT_CODE
                or "KINETO_STALE_HIPKERNELNAMEREF" in result.stderr
            ):
                self.fail(
                    "hipKernelNameRef() was called while the trace was being "
                    "serialized. The kernel name must be captured in the "
                    "hipModuleLaunchKernel callback (0d7ee9e).\n" + result.stderr
                )
            if result.returncode < 0:
                self.fail(
                    f"The profiled process died with signal {-result.returncode} "
                    "while exporting the trace, i.e. the original use-after-free "
                    "still reproduces.\n" + result.stderr
                )
            self.assertEqual(result.returncode, 0, result.stderr)

            with open(trace) as f:
                return json.load(f)["traceEvents"]

    @unittest.skipIf(USES_ROCPROFILER_SDK, "Covers the roctracer backend")
    def test_kernel_name_captured_before_module_unload(self):
        # roctracer attaches the name to the CPU-side launch event.
        events = self._export_guarded_trace(unload_plans=False)

        launches = [e for e in events if e.get("name") == "hipModuleLaunchKernel"]
        self.assertTrue(
            launches, "The FFT workload recorded no hipModuleLaunchKernel activity"
        )
        kernels = [e.get("args", {}).get("kernel") for e in launches]
        self.assertTrue(all(kernels), f"Missing kernel names: {kernels}")
        self.assertTrue(any("fft" in name for name in kernels), kernels)

    @unittest.skipIf(not USES_ROCPROFILER_SDK, "Covers the rocprofiler-sdk backend")
    def test_kernel_name_survives_module_unload_sdk(self):
        # rocprofiler-sdk caches names when the code object is loaded and
        # attaches them to the GPU kernel event, correlated with the launch.
        events = self._export_guarded_trace(unload_plans=True)

        runtime = [e for e in events if e.get("cat") == "cuda_runtime"]
        self.assertTrue(
            any(e["name"] == "hipModuleUnload" for e in runtime),
            "No hipModuleUnload was recorded, the unload path was not exercised",
        )
        launches = [e for e in runtime if e["name"] == "hipModuleLaunchKernel"]
        self.assertTrue(
            launches, "The FFT workload recorded no hipModuleLaunchKernel activity"
        )
        kernel_names = {
            e["args"]["correlation"]: e["name"]
            for e in events
            if e.get("cat") == "kernel"
        }
        names = [kernel_names.get(e["args"]["correlation"]) for e in launches]
        self.assertTrue(
            all(n and n != "<unknown kernel>" for n in names),
            f"Unresolved kernel names: {names}",
        )
        self.assertTrue(any("fft" in n for n in names), names)

if __name__ == "__main__":
    run_tests()
