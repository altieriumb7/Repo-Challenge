#!/usr/bin/env bash
set -euo pipefail
python -m mirror run-scenario --train-dir "$1" --eval-dir "$2" --name "$3" --output-dir "${4:-outputs}"
