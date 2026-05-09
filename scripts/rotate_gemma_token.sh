#!/usr/bin/env bash
# Rotate the gemma-proxy bearer token by redeploying with a fresh value.
#
# Usage: ./scripts/rotate_gemma_token.sh
#
# Just calls deploy_gemma_proxy.sh without an explicit PROXY_TOKEN, which
# generates a new one. The previous token is invalidated when the old proxy
# process is killed.

set -euo pipefail
unset PROXY_TOKEN
exec "$(dirname "${BASH_SOURCE[0]}")/deploy_gemma_proxy.sh"
