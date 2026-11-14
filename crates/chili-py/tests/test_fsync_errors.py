"""Linux syscall fault injection; no real storage failures are induced."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("cc"), reason="Linux LD_PRELOAD and cc required")
def test_first_sync_error_propagates_and_preserves_lock(tmp_path):
    library = tmp_path / "failsync.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(library), str(Path(__file__).with_name("failsync.c")), "-ldl"], check=True)
    env = dict(os.environ, LD_PRELOAD=str(library))
    subprocess.run([sys.executable, "-c", CHILD], env=env, check=True, capture_output=True, text=True)

CHILD = r'''
import ctypes
import errno
import fcntl
import json
import os
import tempfile

from chili import ChiliEngine

probe = ctypes.CDLL(os.environ["LD_PRELOAD"])


def lock_available(path):
    with open(path, "rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
        return True


out = []
for error, failures in [(errno.EIO, 1), (errno.EIO, 2), (errno.ENOSPC, 1), (errno.EDQUOT, 1), (errno.EINVAL, 1), (errno.EINTR, 1), (errno.EINTR, 2)]:
    with tempfile.TemporaryDirectory() as directory:
        path = directory + "/log"
        engine = ChiliEngine(pepper=True)
        try:
            h = engine.open_handle("file://" + path)
            engine.set_var("h", h)
            engine.eval("h (`upd; `trade; 1)")
            before = lock_available(path)
            probe.probe_arm(failures, error)
            try:
                engine.fsync_handle(h)
                result = "success"
            except Exception as exc:
                result = str(exc)
            finally:
                probe.probe_disarm()
            trace = [{"fd": probe.probe_fd(i), "errno": probe.probe_result(i)} for i in range(probe.probe_calls())]
            out.append({"injected_errno": errno.errorcode[error], "failures": failures, "result": result,
                        "trace": trace, "lock_available_before": before, "lock_available_after": lock_available(path)})
        finally:
            engine.shutdown()
for row in out:
    assert not row["lock_available_before"] and not row["lock_available_after"], row
    assert len({x["fd"] for x in row["trace"]}) == 1, row
    if row["injected_errno"] != "EINTR":
        assert row["result"] != "success", row
        assert len(row["trace"]) == 1, row
    else:
        assert row["result"] == "success", row
print(json.dumps(out))

'''
