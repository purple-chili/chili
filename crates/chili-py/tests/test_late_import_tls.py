"""Late-loaded Python extensions must not depend on spare static TLS."""

import os
import platform
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("readelf"), reason="Linux readelf required")
def test_extension_has_no_mandatory_static_tls():
    import chili.engine_state as extension
    from check_wheel_tls import check_elf

    check_elf(extension.__file__)


@pytest.mark.skipif(
    sys.platform != "linux" or platform.libc_ver()[0] != "glibc" or not shutil.which("cc"),
    reason="glibc Linux and a C compiler required",
)
@pytest.mark.parametrize("padding", [512, 1024])
def test_late_import_with_static_tls_pressure(tmp_path, padding):
    source = tmp_path / "pad.c"
    library = tmp_path / "pad.so"
    source.write_text(
        f'static __thread char pad[{padding}] __attribute__((tls_model("initial-exec")));\n'
        'void *tls_pad_address(void) { return pad; }\n'
    )
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(library), str(source)], check=True)
    env = dict(os.environ, POLARS_MAX_THREADS="1")
    env.pop("GLIBC_TUNABLES", None)  # exercise the default loader budget
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(library)], env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


CHILD = r'''
import ctypes
import sys
from concurrent.futures import ThreadPoolExecutor

padding = ctypes.CDLL(sys.argv[1])
padding.tls_pad_address()
import polars as pl
from chili import ChiliEngine

def work(i):
    engine = ChiliEngine(pepper=True, compaction_interval=0)
    try:
        engine.set_var("frame", pl.DataFrame({"n": [i, i + 1]}))
        snapshot = engine.get_var("frame")
        assert snapshot["n"].to_list() == [i, i + 1]
    finally:
        engine.shutdown()
    assert snapshot["n"].to_list() == [i, i + 1]

with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(work, range(8)))
'''
