#!/usr/bin/env bash
# Signs every library of a wheel from scripts/package-python.sh with a Developer ID (hardened runtime, secure timestamp), verifies the signatures, submits the
# libraries to Apple's notary service and waits for the verdict, then writes the re-packed wheel (fresh RECORD hashes) to OUT_DIR.
#
#   DOOMS_CODESIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" DOOMS_NOTARY_PROFILE=dooms scripts/notarize-wheel.sh dist/dooms_cpp-*.whl [OUT_DIR]
#
# DOOMS_NOTARY_PROFILE is a notarytool keychain profile: xcrun notarytool store-credentials dooms --apple-id ... --team-id ... (once, interactively).
# DOOMS_NOTARIZE_SKIP_SUBMIT=1 signs and verifies only (no upload to Apple), for checking the signing setup.
# A wheel is a zip that cannot be stapled: Gatekeeper fetches the notarization ticket of each library online the first time it is loaded.
# Exit codes: 0 notarized (or signed, with SKIP_SUBMIT), 1 a step failed, 2 usage or missing configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${DOOMS_PYTHON:-.venv/bin/python}"   # needs the `wheel` package (as scripts/package-python.sh does)

usage() { echo "usage: DOOMS_CODESIGN_IDENTITY=... DOOMS_NOTARY_PROFILE=... $0 WHEEL [OUT_DIR]   (see the header of this script)" >&2; exit 2; }
[ $# -ge 1 ] && [ $# -le 2 ] || usage
WHEEL="$1"
OUT="${2:-dist/notarized}"
if [ -z "${DOOMS_CODESIGN_IDENTITY:-}" ]; then
    echo "error: DOOMS_CODESIGN_IDENTITY is not set. Set it to your Developer ID Application identity; list them with: security find-identity -v -p codesigning" >&2
    exit 2
fi
SKIP_SUBMIT="${DOOMS_NOTARIZE_SKIP_SUBMIT:-0}"
if [ "$SKIP_SUBMIT" != 1 ] && [ -z "${DOOMS_NOTARY_PROFILE:-}" ]; then
    echo "error: DOOMS_NOTARY_PROFILE is not set. Store credentials once with: xcrun notarytool store-credentials <profile> --apple-id <id> --team-id <team>" \
         "(app-specific password), then set DOOMS_NOTARY_PROFILE=<profile>" >&2
    exit 2
fi
if ! security find-identity -v -p codesigning | grep -qF "\"$DOOMS_CODESIGN_IDENTITY\""; then
    echo "error: no valid code-signing identity named \"$DOOMS_CODESIGN_IDENTITY\" in the keychain (security find-identity -v -p codesigning)" >&2
    exit 2
fi
case "$DOOMS_CODESIGN_IDENTITY" in "Developer ID Application:"*) ;; *)
    echo "error: notarization needs a \"Developer ID Application: ...\" identity, not \"$DOOMS_CODESIGN_IDENTITY\"" >&2; exit 2 ;; esac
[ -f "$WHEEL" ] || { echo "error: no such wheel: $WHEEL (build it with scripts/package-python.sh)" >&2; exit 2; }
case "$WHEEL" in *.whl) ;; *) echo "error: $WHEEL is not a .whl file" >&2; exit 2 ;; esac
"$PY" -c "import wheel" 2>/dev/null || { echo "error: $PY cannot import wheel (uv pip install --python $PY wheel)" >&2; exit 2; }
if [ "$SKIP_SUBMIT" != 1 ]; then
    xcrun notarytool history --keychain-profile "$DOOMS_NOTARY_PROFILE" >/dev/null 2>&1 || {
        echo "error: notarytool cannot use the keychain profile \"$DOOMS_NOTARY_PROFILE\" (missing, or the credentials are rejected); re-run notarytool store-credentials" >&2
        exit 2; }
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/dooms-notarize.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
"$PY" -m wheel unpack "$WHEEL" -d "$WORK/unpacked" >/dev/null
PKG="$(find "$WORK/unpacked" -mindepth 1 -maxdepth 1 -type d)"
LIBS=()
while IFS= read -r -d '' f; do LIBS+=("$f"); done < <(find "$PKG" -type f \( -name '*.dylib' -o -name '*.so' \) -print0)
[ ${#LIBS[@]} -gt 0 ] || { echo "error: $WHEEL contains no libraries to sign" >&2; exit 1; }

echo "signing ${#LIBS[@]} libraries as \"$DOOMS_CODESIGN_IDENTITY\""
for lib in "${LIBS[@]}"; do
    codesign --force --options runtime --timestamp --sign "$DOOMS_CODESIGN_IDENTITY" "$lib"
done
for lib in "${LIBS[@]}"; do
    codesign --verify --strict --verbose=1 "$lib" 2>/dev/null || { echo "error: the signature of $(basename "$lib") does not verify" >&2; exit 1; }
    info="$(codesign -dvv "$lib" 2>&1)"
    grep -qF "Authority=Developer ID Application:" <<<"$info" || { echo "error: $(basename "$lib") is not signed by a Developer ID" >&2; exit 1; }
    grep -q "flags=.*runtime" <<<"$info" || { echo "error: $(basename "$lib") lacks the hardened runtime" >&2; exit 1; }
    grep -q "^Timestamp=" <<<"$info" || { echo "error: $(basename "$lib") has no secure timestamp" >&2; exit 1; }
done
echo "signatures verified (Developer ID, hardened runtime, timestamp)"

if [ "$SKIP_SUBMIT" != 1 ]; then
    ZIP="$WORK/notarize.zip"
    ( cd "$PKG" && ditto -c -k --sequesterRsrc . "$ZIP" )
    echo "submitting to the notary service (this waits for the verdict, usually minutes)"
    RESULT="$WORK/submit.plist"
    xcrun notarytool submit "$ZIP" --keychain-profile "$DOOMS_NOTARY_PROFILE" --wait --output-format plist > "$RESULT" || true
    STATUS="$(/usr/libexec/PlistBuddy -c 'Print :status' "$RESULT" 2>/dev/null || echo unknown)"
    ID="$(/usr/libexec/PlistBuddy -c 'Print :id' "$RESULT" 2>/dev/null || echo unknown)"
    if [ "$STATUS" != "Accepted" ]; then
        echo "error: notarization status \"$STATUS\" (submission $ID). Details: xcrun notarytool log $ID --keychain-profile $DOOMS_NOTARY_PROFILE" >&2
        exit 1
    fi
    echo "notarized: submission $ID accepted"
fi

mkdir -p "$OUT"
"$PY" -m wheel pack "$PKG" -d "$OUT" >/dev/null
ls -la "$OUT"/*.whl
[ "$SKIP_SUBMIT" = 1 ] && echo "note: signed but NOT notarized (DOOMS_NOTARIZE_SKIP_SUBMIT=1); do not publish this wheel"
exit 0
