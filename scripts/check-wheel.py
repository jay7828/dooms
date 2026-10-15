#!/usr/bin/env python3
"""Checks a wheel from scripts/package-python.sh before it is published.

    python3 scripts/check-wheel.py dist/dooms_cpp-X.Y.Z-py3-none-macosx_14_0_arm64.whl

- the platform tag is macosx_<X>_<Y>_arm64 with X.Y the newest minimum macOS (LC_BUILD_VERSION minos) of the bundled libraries, and every library is arm64 only;
- no model weights (GGUF, safetensors, Core ML packages, checkpoints) are inside;
- the metadata names dooms-cpp at the version in VERSION, Apache-2.0, Python >= 3.9, the `dooms` console script, and the licence files (ours and llama.cpp's);
- no library refers to a build tree (only @rpath / @loader_path / system paths).
Exit codes: 0 ok, 1 a check failed (each failure listed), 2 usage.
"""
import os
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DISTRIBUTION = "dooms_cpp"
MODEL_FILES = re.compile(r"\.(gguf|safetensors|mlpackage|mlmodelc|pt|pth|onnx|npz|bin)(/|$)", re.I)
SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/", "@rpath/", "@loader_path/")


def minos(path):
    out = subprocess.run(["otool", "-l", path], capture_output=True, text=True, check=True).stdout
    m = re.search(r"cmd LC_BUILD_VERSION.*?minos (\d+)\.(\d+)", out, re.S)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main(argv):
    if len(argv) != 1 or not argv[0].endswith(".whl"):
        print(__doc__, file=sys.stderr)
        return 2
    wheel = argv[0]
    if not os.path.isfile(wheel):
        print(f"error: no such wheel: {wheel}", file=sys.stderr)
        return 2
    with open(os.path.join(ROOT, "VERSION")) as f:
        version = f.read().strip()
    bad = []
    m = re.fullmatch(rf"{DISTRIBUTION}-(?P<v>[^-]+)-py3-none-macosx_(?P<maj>\d+)_(?P<min>\d+)_arm64\.whl", os.path.basename(wheel))
    if not m:
        return print(f"error: {os.path.basename(wheel)} is not named {DISTRIBUTION}-<version>-py3-none-macosx_<X>_<Y>_arm64.whl", file=sys.stderr) or 1
    if m.group("v") != version:
        bad.append(f"version {m.group('v')} in the file name, {version} in VERSION")
    with zipfile.ZipFile(wheel) as z, tempfile.TemporaryDirectory() as tmp:
        names = z.namelist()
        bad += [f"model file inside: {n}" for n in names if MODEL_FILES.search(n)]
        dist_info = f"{DISTRIBUTION}-{version}.dist-info/"
        meta = z.read(dist_info + "METADATA").decode() if dist_info + "METADATA" in names else ""
        for want in (f"Name: dooms-cpp\n", f"Version: {version}\n", "License: Apache-2.0\n", "Requires-Python: >=3.9\n", "Description-Content-Type: text/markdown\n"):
            if want not in meta:
                bad.append(f"METADATA lacks {want.strip()!r}")
        for lic in ("LICENSE", "NOTICE", "LICENSE-llama.cpp"):
            if dist_info + lic not in names:
                bad.append(f"licence file {lic} missing from {dist_info}")
        entry = z.read(dist_info + "entry_points.txt").decode() if dist_info + "entry_points.txt" in names else ""
        if "dooms = dooms.cli:main" not in entry:
            bad.append("the `dooms` console script is not declared")
        libs = [n for n in names if n.endswith(".dylib")]
        if not any(n.endswith("/libdooms_c.dylib") for n in libs):
            bad.append("libdooms_c.dylib is missing")
        newest = (11, 0)  # arm64 macOS starts at 11
        for n in libs:
            path = z.extract(n, tmp)
            archs = subprocess.run(["lipo", "-archs", path], capture_output=True, text=True).stdout.split()
            if archs != ["arm64"]:
                bad.append(f"{n}: architectures {archs}, expected arm64 only")
            v = minos(path)
            if v is None:
                bad.append(f"{n}: no LC_BUILD_VERSION")
            else:
                newest = max(newest, v)
            for line in subprocess.run(["otool", "-L", path], capture_output=True, text=True, check=True).stdout.splitlines()[1:]:
                dep = line.strip().split(" (")[0]
                if not dep.startswith(SYSTEM_PREFIXES):
                    bad.append(f"{n}: depends on {dep}")
            rpaths = re.findall(r"cmd LC_RPATH.*?path (\S+)", subprocess.run(["otool", "-l", path], capture_output=True, text=True, check=True).stdout, re.S)
            bad += [f"{n}: rpath {r}" for r in rpaths if r != "@loader_path"]
        tag = (int(m.group("maj")), int(m.group("min")))
        if tag != newest:
            bad.append(f"platform tag says macOS {tag[0]}.{tag[1]}, the libraries need {newest[0]}.{newest[1]}")
    if bad:
        print(f"{wheel}: {len(bad)} problem(s)\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print(f"{os.path.basename(wheel)}: ok ({len(libs)} libraries, macOS {newest[0]}.{newest[1]}+, arm64, no model files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
