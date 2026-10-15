#!/usr/bin/env bash
# Builds a self-contained wheel of the Python binding: dooms/*.py + dooms/lib/{libdooms_c, libllama, libggml*}.dylib, every library relocated to @loader_path so the
# wheel works from any install location (no reference to the build tree remains). The models are NOT included (licenses, ADR 0002): point the classes at them.
# Usage: scripts/package-python.sh [OUT_DIR]      (default OUT_DIR: dist/)     Needs the .venv with setuptools and wheel (uv pip install --python .venv/bin/python wheel).
set -euo pipefail
cd "$(dirname "$0")/.."
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

mkdir -p "$OUT"
( cd "$STAGE" && "../../$PY" setup.py -q bdist_wheel --dist-dir "$(cd ../.. && pwd)/$OUT" )
ls -la "$OUT"/dooms-*.whl
