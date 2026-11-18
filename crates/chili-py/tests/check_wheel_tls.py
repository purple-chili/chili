"""Reject mandatory static TLS in Linux extension wheels, without importing them."""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile


def check_elf(library):
    library = Path(library)
    with library.open("rb") as stream:
        header = stream.read(20)
    if header[:4] != b"\x7fELF":
        raise ValueError(f"{library}: expected an ELF extension")
    machine = int.from_bytes(header[18:20], "little" if header[5] == 1 else "big")
    if machine not in (62, 183):  # AMD64 and AArch64
        raise ValueError(f"{library}: unsupported ELF machine {machine}")
    relocations = subprocess.check_output(["readelf", "-rW", str(library)], text=True)
    dynamic = subprocess.check_output(["readelf", "-dW", str(library)], text=True)
    forbidden = re.findall(r"\bR_(?:X86_64_TPOFF64|AARCH64_TLS_TPREL(?:64)?)\b", relocations)
    if forbidden or "STATIC_TLS" in dynamic:
        raise ValueError(f"{library}: mandatory static TLS: {sorted(set(forbidden))}")


def check_wheel(wheel):
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as directory:
        extensions = [n for n in archive.namelist() if n.endswith("/engine_state.abi3.so")]
        if len(extensions) != 1:
            raise ValueError(f"{wheel}: expected one engine_state extension")
        library = Path(directory) / "engine_state.abi3.so"
        library.write_bytes(archive.read(extensions[0]))
        check_elf(library)
    print(f"{wheel}: no mandatory static TLS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    for wheel in parser.parse_args().wheels:
        check_wheel(wheel)
