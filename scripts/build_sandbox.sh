#!/usr/bin/env bash
# Build the sandbox image. Behind a TLS-intercepting proxy, pass its CA bundle:
#   PIP_CERT_FILE=/path/to/ca.crt scripts/build_sandbox.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=$(python3 -c "from hmas.sandbox import DEFAULT_IMAGE; print(DEFAULT_IMAGE)")
EXTRA=()
if [ -n "${PIP_CERT_FILE:-}" ]; then EXTRA+=(--secret "id=pip_cert,src=$PIP_CERT_FILE" --network host); fi
docker build "${EXTRA[@]}" -t "$IMAGE" -f docker/sandbox.Dockerfile docker
echo "built $IMAGE"
