#!/bin/sh
set -eu

python3 /opt/tinfoil/seed-autotune.py
exec /opt/tinfoil/inference-sidecar "$@"
