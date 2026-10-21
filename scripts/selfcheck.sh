#!/usr/bin/env bash
# Proves the gate rejects wrong code: each mutant is applied to a copy of the tree and
# scripts/check.sh must FAIL on it. Then the oracle as its own stand-in must pass the oracle part.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
mutants=(
  "src/gguf/file.mojo|                    raise Error(\"gguf: \", path, \": duplicate key \", key)|                    pass|duplicate keys accepted"
  "src/gguf/file.mojo|            off += pad_to(t.nbytes, self.alignment)|            off += t.nbytes|tensor offsets not padded"
  "src/gguf/file.mojo|return v - (1 << bits) if t in [1, 3, 5]|return v - (1 << bits) if t in [1, 3]|int32 not sign-extended"
  "src/gguf/file.mojo|        var at = self.kvs[i].start + 12|        var at = self.kvs[i].start + 8|string array offset"
  "src/gguf/file.mojo|84, 110, 144, 176|84, 110, 142, 176|Q4_K block size"
  "src/gguf/file.mojo|                self.vals[i] = v^\n                return|                pass|re-added key appended"
  "src/gguf/file.mojo|            if self.data_offset + t.offset + t.nbytes > self.size:|            if False:|tensor past end of file accepted"
)
for m in "${mutants[@]}"; do
  IFS='|' read -r file from to why <<< "$m"
  d=$ROOT/.work/mutant; rm -rf "$d"; mkdir -p "$d"
  cp -r src tests tools scripts examples "$d"; ln -s "$ROOT/.venv" "$d/.venv"; mkdir -p "$d/.work"; ln -s "$ROOT/.work/fx" "$d/.work/fx"
  python3 -I -c 'import sys; p,a,b=sys.argv[1:]; a=a.replace("\\n","\n"); b=b.replace("\\n","\n"); s=open(p).read(); assert a in s, "mutant does not apply: "+a; open(p,"w").write(s.replace(a,b,1))' "$d/$file" "$from" "$to" \
    || { echo "FAIL selfcheck: mutant does not apply ($why)"; exit 1; }
  if "$d/scripts/check.sh" > "$d.log" 2>&1; then echo "FAIL selfcheck: gate passed a mutant ($why)"; exit 1; fi
  echo "selfcheck: rejected ($why): $(grep -m1 FAIL "$d.log")"
done
CASES="uv run --no-project -q --with gguf==0.19.0 python -I tools/oracle.py" scripts/check.sh > .work/oracle-self.log 2>&1 || { echo "FAIL selfcheck: oracle fails its own gate"; tail -3 .work/oracle-self.log; exit 1; }
echo "PASS selfcheck"
