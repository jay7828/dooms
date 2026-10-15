#!/usr/bin/env bash
# Builds a self-contained wheel of the Python binding: dooms/*.py + dooms/lib/{libdooms_c, libllama, libggml*}.dylib, every library relocated to @loader_path so the
# wheel works from any install location (no reference to the build tree remains). The models are NOT included (licenses, ADR 0002): point the classes at them.
# Usage: scripts/package-python.sh [OUT_DIR]      (default OUT_DIR: dist/)     Needs the .venv with setuptools and wheel (uv pip install --python .venv/bin/python wheel).
# The platform tag is the newest macOS any bundled library requires (LC_BUILD_VERSION minos): configure with -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 for a wheel that installs on
# macOS 14+, otherwise it is tagged with the build machine's macOS. Release steps (signing, notarization, PyPI): docs/RELEASING.md.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${1:-dist}"
BUILD_DIR="${DOOMS_BUILD_DIR:-build}"
if [ -x .venv/bin/cmake ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
PY=.venv/bin/python

cmake -B "$BUILD_DIR" >/dev/null
cmake --build "$BUILD_DIR" --target dooms_c -j
LIBDIR="$BUILD_DIR/bin"
[ -f "$BUILD_DIR/libdooms_c.dylib" ] || { echo "error: $BUILD_DIR/libdooms_c.dylib was not built"; exit 1; }

STAGE="$BUILD_DIR/python-package"
rm -rf "$STAGE"
mkdir -p "$STAGE/dooms/lib"
cp python/dooms/*.py "$STAGE/dooms/"
cp python/setup.py "$STAGE/setup.py"
cp VERSION "$STAGE/VERSION"
cp python/README.md "$STAGE/README.md"   # the PyPI description
# licences shipped in the wheel's metadata: ours (root of the public tree, scripts/public/ here) and llama.cpp/ggml's (MIT; their libraries are bundled)
for f in LICENSE NOTICE; do
    if [ -f "$f" ]; then cp "$f" "$STAGE/$f"; elif [ -f "scripts/public/$f" ]; then cp "scripts/public/$f" "$STAGE/$f"; else echo "error: no $f file"; exit 1; fi
done
LLAMA_SRC="$(awk -F= '/^FETCHCONTENT_SOURCE_DIR_LLAMA_CPP:PATH=/{print $2}' "$BUILD_DIR/CMakeCache.txt")"
[ -n "$LLAMA_SRC" ] || LLAMA_SRC="$BUILD_DIR/_deps/llama_cpp-src"
[ -f "$LLAMA_SRC/LICENSE" ] || { echo "error: llama.cpp's LICENSE not found in $LLAMA_SRC"; exit 1; }
cp "$LLAMA_SRC/LICENSE" "$STAGE/LICENSE-llama.cpp"

# dooms_c plus everything it needs from the build's library directory (real files, not the symlinks)
cp "$BUILD_DIR/libdooms_c.dylib" "$STAGE/dooms/lib/"
copy_deps() {  # copy_deps <dylib>: copy each @rpath/<name> dependency found in $LIBDIR (recursively)
    local lib="$1" dep name
    for dep in $(otool -L "$lib" | awk '/@rpath\//{print $1}'); do
        name="${dep#@rpath/}"
        [ "$name" = "$(basename "$lib")" ] && continue
        if [ ! -e "$STAGE/dooms/lib/$name" ]; then
            [ -e "$LIBDIR/$name" ] || { echo "error: dependency $name of $lib not found in $LIBDIR"; exit 1; }
            cp -L "$LIBDIR/$name" "$STAGE/dooms/lib/$name"
            copy_deps "$STAGE/dooms/lib/$name"
        fi
    done
}
copy_deps "$STAGE/dooms/lib/libdooms_c.dylib"
# ggml loads its backends (cpu, blas, metal) at run time from the directory of libggml; they are not link-time dependencies
for b in "$LIBDIR"/libggml-*.0.dylib; do cp -L "$b" "$STAGE/dooms/lib/$(basename "$b")"; done

# relocate: no absolute rpath left, @loader_path instead; the dylibs are ad-hoc re-signed because install_name_tool invalidates the signature
for lib in "$STAGE"/dooms/lib/*.dylib; do
    for rp in $(otool -l "$lib" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}'); do install_name_tool -delete_rpath "$rp" "$lib"; done
    install_name_tool -add_rpath @loader_path "$lib"
    codesign --force --sign - "$lib" >/dev/null 2>&1
done
if otool -l "$STAGE"/dooms/lib/*.dylib | grep -q "$PWD"; then echo "error: a library still refers to the build tree"; exit 1; fi

# platform tag: arm64 only, and the newest minimum macOS of any bundled library
MINOS=""
for lib in "$STAGE"/dooms/lib/*.dylib; do
    archs="$(lipo -archs "$lib")"
    [ "$archs" = "arm64" ] || { echo "error: $(basename "$lib") is built for '$archs'; the wheel is arm64 only"; exit 1; }
    v="$(otool -l "$lib" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')"
    [ -n "$v" ] || { echo "error: $(basename "$lib") has no LC_BUILD_VERSION (minimum macOS unknown)"; exit 1; }
    if [ -z "$MINOS" ] || [ "$(printf '%s\n%s\n' "$MINOS" "$v" | sort -V | tail -1)" = "$v" ]; then MINOS="$v"; fi
done
MAJOR="${MINOS%%.*}"; REST="${MINOS#*.}"; MINOR="${REST%%.*}"
[ "$MAJOR" -ge 11 ] || { MAJOR=11; MINOR=0; }   # arm64 macOS starts at 11
export DOOMS_WHEEL_PLATFORM="macosx_${MAJOR}_${MINOR}_arm64"
echo "platform tag: $DOOMS_WHEEL_PLATFORM (the libraries need macOS $MINOS)"

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"   # absolute: setup.py runs from the staging directory
VERSION_STR="$(cat VERSION)"
WHEEL="$OUT/dooms_cpp-$VERSION_STR-py3-none-$DOOMS_WHEEL_PLATFORM.whl"
rm -f "$WHEEL"
( cd "$STAGE" && "$ROOT/$PY" setup.py -q bdist_wheel --dist-dir "$OUT" >/dev/null )
[ -f "$WHEEL" ] || { echo "error: expected $WHEEL"; exit 1; }
# no model weights in the wheel (licences, docs/adr/0002)
if unzip -Z1 "$WHEEL" | grep -Ei '\.(gguf|safetensors|mlpackage|mlmodelc|pt|onnx|npz)(/|$)'; then echo "error: model files in $WHEEL"; exit 1; fi
ls -la "$WHEEL"
