#!/usr/bin/env bash
# CI entry point, same on a GitHub runner and a local Ubuntu VM: install, then the gate.
set -euo pipefail
cd "$(dirname "$0")/.."
v=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml); c=$(sed -n 's/^- \[v\([0-9.]*\)\].*/\1/p' CHANGELOG.md | head -1)
[ "$v" = "$c" ] || { echo "FAIL version: pyproject $v vs newest CHANGELOG entry $c"; exit 1; }
command -v uv >/dev/null || { echo "FAIL setup: uv not installed"; exit 1; }
uv sync || { echo "FAIL setup: uv sync"; exit 1; }
python3 -c 'import platform; print("oracle python", platform.python_version())'
scripts/check.sh
