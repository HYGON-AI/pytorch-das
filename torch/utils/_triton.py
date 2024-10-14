# mypy: allow-untyped-defs
import functools
import hashlib
import os


@functools.lru_cache(None)
def has_triton_package() -> bool:
    try:
        import triton

        return triton is not None
    except ImportError:
        return False


@functools.lru_cache(None)
def has_triton() -> bool:
    from torch._dynamo.device_interface import get_interface_for_device

    def cuda_extra_check(device_interface):
        return device_interface.Worker.get_device_properties().major >= 7

    def _return_true(device_interface):
        return True

    triton_supported_devices = {"cuda": cuda_extra_check, "xpu": _return_true}

    def is_device_compatible_with_triton():
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    return is_device_compatible_with_triton() and has_triton_package()


@functools.lru_cache(None)
def triton_backend_hash():
    from triton.common.backend import get_backend, get_cuda_version_key

    import torch

    if torch.version.hip:
        # Does not work with ROCm
        return None

    if not torch.cuda.is_available():
        return None

    backend = get_backend("cuda")
    if backend is None:
        return get_cuda_version_key()
    else:
        return backend.get_version_key()


@functools.lru_cache
def triton_key():
    import pkgutil

    import triton

    TRITON_PATH = os.path.dirname(os.path.abspath(triton.__file__))
    contents = []
    # This is redundant. Doing it to be consistent with upstream.
    # frontend
    with open(os.path.join(TRITON_PATH, "compiler", "compiler.py"), "rb") as f:
        contents += [hashlib.sha256(f.read()).hexdigest()]

    # compiler
    compiler_path = os.path.join(TRITON_PATH, "compiler")
    backends_path = os.path.join(TRITON_PATH, "compiler", "backends")
    for lib in pkgutil.iter_modules([compiler_path, backends_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:  # type: ignore[call-arg, union-attr, arg-type]
            contents += [hashlib.sha256(f.read()).hexdigest()]
    # backend
    libtriton_hash = hashlib.sha256()
    with open(os.path.join(TRITON_PATH, "_C/libtriton.so"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, "language")
    for lib in pkgutil.iter_modules([language_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:  # type: ignore[call-arg, union-attr, arg-type]
            contents += [hashlib.sha256(f.read()).hexdigest()]
    from triton import __version__

    return f"{__version__}" + "-".join(contents)


@functools.lru_cache(None)
def triton_hash_with_backend():
    import torch

    if torch.version.hip:
        # Does not work with ROCm
        return None

    from triton.compiler.compiler import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}"

    # Hash is upper case so that it can't contain any Python keywords.
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()


def dtype_to_string(dtype):
    if dtype.name.startswith("fp"):
        suffix = "float" + dtype.name[2:]
    elif dtype.name.startswith("bf"):
        suffix = "bfloat" + dtype.name[2:]
    else:
        suffix = dtype.name
    return "triton.language." + suffix


def patch_triton_dtype_repr():
    import triton

    # Hack to get triton dtype repr to produce an evaluatable expression
    # triton.language.float32 emits triton.language.fp32 which does not
    # exist
    # REMOVE when https://github.com/openai/triton/pull/3342 lands
    triton.language.dtype.__repr__ = lambda self: dtype_to_string(self)
