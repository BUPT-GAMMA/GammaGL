#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export TL_BACKEND=torch
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

python "${ROOT_DIR}/examples/coed/coed_trainer.py" "$@"
