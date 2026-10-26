#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Prefer the system Node installation. Codex's bundled runtime is an optional fallback.
if ! command -v node >/dev/null || [[ "$(node -p 'Number(process.versions.node.split(".")[0])')" -lt 22 ]]; then
  bundled_node="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin"
  if [[ -x "$bundled_node/node" ]]; then
    export PATH="$bundled_node:$PATH"
  else
    echo "Install Node.js 22.13 or newer before starting HydroSense." >&2
    exit 1
  fi
fi
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  echo "Create .venv and install the project first; see README.md." >&2
  exit 1
fi
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$project_root/.venv/bin/python" -m hydrosense.service &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true' EXIT INT TERM
cd "$project_root/webapp"
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
