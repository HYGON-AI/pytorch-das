from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

from setuptools import distutils  # type: ignore[import]


UNKNOWN = "Unknown"
RELEASE_PATTERN = re.compile(r"/v[0-9]+(\.[0-9]+)*(-rc[0-9]+)?/")


def get_sha(pytorch_root: str | Path) -> str:
    try:
        rev = None
        if os.path.exists(os.path.join(pytorch_root, ".git")):
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=pytorch_root
            )
        elif os.path.exists(os.path.join(pytorch_root, ".hg")):
            rev = subprocess.check_output(
                ["hg", "identify", "-r", "."], cwd=pytorch_root
            )
        if rev:
            return rev.decode("ascii").strip()
    except Exception:
        pass
    return UNKNOWN


def get_tag(pytorch_root: str | Path) -> str:
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact"],
            cwd=pytorch_root,
            encoding="ascii",
            capture_output=True,
        ).stdout.strip()
        if RELEASE_PATTERN.match(tag):
            return tag
        else:
            return UNKNOWN
    except Exception:
        return UNKNOWN

def get_abi():
    try:
        command = "echo '#include <string>' | gcc -x c++ -E -dM - | fgrep _GLIBCXX_USE_CXX11_ABI"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        abi = "abi" + output.split(" ")[-1]
        return abi
    except Exception:
        return 'abiUnknown'

def get_torch_version(sha: str | None = None) -> str:
    pytorch_root = Path(__file__).absolute().parent.parent
    version = open(pytorch_root / "version.txt").read().strip()

    if os.getenv("PYTORCH_BUILD_VERSION"):
        assert os.getenv("PYTORCH_BUILD_NUMBER") is not None
        build_number = int(os.getenv("PYTORCH_BUILD_NUMBER", ""))
        version = os.getenv("PYTORCH_BUILD_VERSION", "")
        if build_number > 1:
            version += ".post" + str(build_number)
    elif sha != UNKNOWN:
        pass
    #    if sha is None:
    #        sha = get_sha(pytorch_root)
    #    version += "+git" + sha[:7]
    return version

def get_hcu_version(sha: Optional[str] = None) -> str:
    pytorch_root = Path(__file__).parent.parent
    hcu_version = open(pytorch_root / "version.txt", "r").read().strip()

    if sha != UNKNOWN:
        if sha is None:
            sha = get_sha(pytorch_root)
        #hcu_version += "+git" + sha[:7]
    hcu_version += "+das.opt1"
    #abi
    #hcu_version += "." + get_abi()
    if os.getenv('PYTORCH_CUSTOM_SUFFIX'):
        hcu_version += "." + str(os.getenv('PYTORCH_CUSTOM_SUFFIX'))
    if os.getenv("ROCM_PATH"):
        rocm_path = os.getenv('ROCM_PATH', "")
        rocm_version_path = os.path.join(rocm_path, '.info', "rocm_version")
        with open(rocm_version_path, 'r',encoding='utf-8') as file:
            lines = file.readlines()
        rocm_version=lines[0].replace(".", "")
        #rocm_version=lines[0][:-2].replace(".", "")
        hcu_version += ".dtk" + rocm_version
    # if os.getenv('PYTORCH_CUSTOM_SUFFIX'):
    #     hcu_version += "." + str(os.getenv('PYTORCH_CUSTOM_SUFFIX'))
    return hcu_version

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate torch/version.py from build and environment metadata."
    )
    parser.add_argument(
        "--is-debug",
        "--is_debug",
        type=distutils.util.strtobool,
        help="Whether this build is debug mode or not.",
    )
    parser.add_argument("--cuda-version", "--cuda_version", type=str)
    parser.add_argument("--hip-version", "--hip_version", type=str)
    parser.add_argument("--dtk-version", "--dtk_version", type=str)
    parser.add_argument("--xpu-version", "--xpu_version", type=str)

    args = parser.parse_args()

    assert args.is_debug is not None
    args.cuda_version = None if args.cuda_version == "" else args.cuda_version
    args.hip_version = None if args.hip_version == "" else args.hip_version
    args.dtk_version = None if args.dtk_version == "" else args.dtk_version
    args.xpu_version = None if args.xpu_version == "" else args.xpu_version

    pytorch_root = Path(__file__).parent.parent
    version_path = pytorch_root / "torch" / "version.py"
    # Attempt to get tag first, fall back to sha if a tag was not found
    tagged_version = get_tag(pytorch_root)
    sha = get_sha(pytorch_root)
    if tagged_version == UNKNOWN:
        version = get_torch_version(sha)
    else:
        version = tagged_version

    hcu_version = get_hcu_version(sha)

    with open(version_path, "w") as f:
        f.write("from typing import Optional\n\n")
        f.write("__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']\n")
        f.write(f"__version__ = '{version}'\n")
        # NB: This is not 100% accurate, because you could have built the
        # library code with DEBUG, but csrc without DEBUG (in which case
        # this would claim to be a release build when it's not.)
        f.write(f"debug = {repr(bool(args.is_debug))}\n")
        f.write(f"cuda: Optional[str] = {repr(args.cuda_version)}\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write(f"hip: Optional[str] = {repr(args.hip_version)}\n")
        f.write(f"__hcu_version__ = '{hcu_version}'\n")
        f.write(f"dtk = {repr(args.dtk_version)}\n")
        f.write(f"xpu: Optional[str] = {repr(args.xpu_version)}\n")
