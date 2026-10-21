#!/usr/bin/env bash
# The gate. Fixtures from tools/fixtures_oracle.py: files gguf-py's GGUFWriter built from
# tests/specs/*.txt, and corrupted copies. For every file, $CASES read must print what gguf-py's
# GGUFReader prints (tools/oracle.py), "error" included. Every readable file re-emitted through
# Writer (copy) and every spec built through the typed Writer calls (build) must be byte-identical
# to what GGUFWriter writes. The ggml type table must equal gguf-py's. Then real files (local
# only, when present), the API test and the example.
# CASES="$O tools/oracle.py" proves the gate with the oracle as its own stand-in.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .work
M=.venv/bin/mojo
O="uv run --no-project -q --with gguf==0.19.0 python -I"
CASES=${CASES:-.work/cases}
if [[ $CASES == .work/cases ]]; then
  $M build -I src tests/cases.mojo -o .work/cases 2>.work/build.err || { echo "FAIL build: tests/cases.mojo"; cat .work/build.err; exit 1; }
fi
F=.work/fx
[[ -f $F/many.gguf ]] || $O tools/fixtures_oracle.py $F || { echo "FAIL setup: fixtures"; exit 1; }
[[ $($CASES types) == "$($O tools/oracle.py types)" ]] || { echo "FAIL oracle: ggml type table differs from gguf-py"; exit 1; }
echo "types: $($CASES types | wc -l) ggml types, block and byte sizes identical"
nr=0; ne=0; nc=0
for f in $F/*.gguf; do
  got=$($CASES read "$f" 2>&1) || got="crash (exit $?): ${got: -200}"; want=$($O tools/oracle.py read "$f")
  [[ $got == "$want" ]] || { echo "FAIL oracle: read $(basename "$f")"; diff <(echo "$want") <(echo "$got") | head -8; exit 1; }
  if [[ $got == error ]]; then ne=$((ne + 1)); continue; fi
  nr=$((nr + 1))
  rm -f .work/c.impl
  $CASES copy "$f" .work/c.impl > /dev/null
  cmp -s .work/c.impl "$f" || { echo "FAIL writer: copy of $(basename "$f") differs from the GGUFWriter file"; exit 1; }
  nc=$((nc + 1))
done
echo "oracle: $nr files read identically, $ne rejected as gguf-py rejects them"
echo "writer: $nc copies byte-identical to the GGUFWriter originals"
nb=0
for s in tests/specs/*.txt; do
  rm -f .work/b.impl .work/b.ref
  $CASES build "$s" .work/b.impl > /dev/null; $O tools/oracle.py build "$s" .work/b.ref > /dev/null
  cmp -s .work/b.impl .work/b.ref || { echo "FAIL writer: build $(basename "$s") differs from GGUFWriter"; exit 1; }
  nb=$((nb + 1))
done
echo "writer: $nb specs built through the typed calls, byte-identical to GGUFWriter"
nl=0
for f in ${REAL:-$HOME/Models/RegesCore-1.0-35/mmproj-F16.gguf}; do
  [[ -f $f ]] || continue
  [[ $($CASES read "$f") == "$($O tools/oracle.py read "$f")" ]] || { echo "FAIL oracle: real file $f"; exit 1; }
  nl=$((nl + 1))
done
echo "real: $nl local files read identically${REAL:+ ($REAL)}"
if [[ $CASES == .work/cases ]]; then
  $M run -I src tests/api.mojo || { echo "FAIL api: tests/api.mojo"; exit 1; }
  $M run -I src tests/coexist.mojo || { echo "FAIL coexist: tests/coexist.mojo (a libc declaration clashes with the stdlib)"; exit 1; }
  $M run -I src examples/quickstart.mojo > /dev/null || { echo "FAIL example: examples/quickstart.mojo"; exit 1; }
fi
echo "PASS all"
