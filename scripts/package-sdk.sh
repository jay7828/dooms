#!/usr/bin/env bash
# Builds the C/C++ SDK tarball: include/dooms/{dooms_c.h,dooms.hpp}, lib/libdooms_c.dylib with libllama/libggml* (relocated to @loader_path), lib/cmake/dooms (find_package),
# lib/pkgconfig/dooms.pc, bin/dooms-cli and the examples. Models are NOT included (licenses, ADR 0002).
# Usage: scripts/package-sdk.sh [OUT_DIR]      (default OUT_DIR: dist/)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-dist}"
BUILD_DIR="${DOOMS_BUILD_DIR:-build}"
if [ -x .venv/bin/cmake ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
VERSION="$(tr -d '[:space:]' < VERSION)"
NAME="dooms-sdk-$VERSION-macos-arm64"

cmake -B "$BUILD_DIR" >/dev/null
cmake --build "$BUILD_DIR" --target dooms_c dooms-cli -j
LIBDIR="$BUILD_DIR/bin"
STAGE="$BUILD_DIR/sdk/$NAME"
rm -rf "$BUILD_DIR/sdk"
mkdir -p "$STAGE"/{include/dooms,lib/cmake/dooms,lib/pkgconfig,bin,share/doc/dooms,share/dooms/examples}
cp include/dooms/dooms_c.h include/dooms/dooms.hpp "$STAGE/include/dooms/"
cp -R examples/cpp examples/python "$STAGE/share/dooms/examples/"
for f in README.md LICENSE NOTICE; do [ -f "$f" ] && cp "$f" "$STAGE/share/doc/dooms/"; done

cp "$BUILD_DIR/libdooms_c.dylib" "$STAGE/lib/"
copy_deps() {
    local lib="$1" dep name src
    for dep in $(otool -L "$lib" | awk '/@rpath\//{print $1}'); do
        name="${dep#@rpath/}"
        [ "$name" = "$(basename "$lib")" ] && continue
        if [ ! -e "$STAGE/lib/$name" ]; then
            src="$LIBDIR/$name"; [ -e "$src" ] || src="$BUILD_DIR/$name"
            [ -e "$src" ] || { echo "error: dependency $name of $lib not found in $LIBDIR or $BUILD_DIR"; exit 1; }
            cp -L "$src" "$STAGE/lib/$name"
            copy_deps "$STAGE/lib/$name"
        fi
    done
}
copy_deps "$STAGE/lib/libdooms_c.dylib"
for b in "$LIBDIR"/libggml-*.0.dylib; do cp -L "$b" "$STAGE/lib/$(basename "$b")"; done   # backends are loaded at run time from the directory of libggml
cp "$BUILD_DIR/dooms-cli" "$STAGE/bin/dooms-cli"
copy_deps "$STAGE/bin/dooms-cli"

relocate() {  # relocate <file> <rpath>
    local f="$1" rp
    for rp in $(otool -l "$f" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}'); do install_name_tool -delete_rpath "$rp" "$f"; done
    install_name_tool -add_rpath "$2" "$f"
    codesign --force --sign - "$f" >/dev/null 2>&1
}
for lib in "$STAGE"/lib/*.dylib; do relocate "$lib" @loader_path; done
relocate "$STAGE/bin/dooms-cli" @loader_path/../lib
if otool -l "$STAGE"/lib/*.dylib "$STAGE/bin/dooms-cli" | grep -q "$PWD"; then echo "error: a binary still refers to the build tree"; exit 1; fi

cat > "$STAGE/lib/cmake/dooms/dooms-config.cmake" <<'CMAKE'
# find_package(dooms): imports dooms::dooms (the C API library; headers <dooms/dooms_c.h> and <dooms/dooms.hpp>).
get_filename_component(_dooms_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT TARGET dooms::dooms)
  add_library(dooms::dooms SHARED IMPORTED)
  set_target_properties(dooms::dooms PROPERTIES
    IMPORTED_LOCATION "${_dooms_prefix}/lib/libdooms_c.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "${_dooms_prefix}/include"
    INTERFACE_COMPILE_FEATURES cxx_std_17)
endif()
set(dooms_FOUND TRUE)
unset(_dooms_prefix)
CMAKE
cat > "$STAGE/lib/cmake/dooms/dooms-config-version.cmake" <<CMAKE
set(PACKAGE_VERSION "$VERSION")
string(REGEX MATCH "^[0-9]+\\\\.[0-9]+" _have "\${PACKAGE_VERSION}")
string(REGEX MATCH "^[0-9]+\\\\.[0-9]+" _want "\${PACKAGE_FIND_VERSION}")
if(PACKAGE_FIND_VERSION AND NOT _have STREQUAL _want AND PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
endif()
CMAKE
cat > "$STAGE/lib/pkgconfig/dooms.pc" <<PC
prefix=\${pcfiledir}/../..
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: dooms
Description: dooms.cpp inference engine (Apple Silicon): LLM, VLM, VLA
Version: $VERSION
Libs: -L\${libdir} -ldooms_c -Wl,-rpath,\${libdir}
Cflags: -I\${includedir}
PC

mkdir -p "$OUT"
tar -C "$BUILD_DIR/sdk" -czf "$OUT/$NAME.tar.gz" "$NAME"
ls -la "$OUT/$NAME.tar.gz"
