#!/bin/bash
# Run from the server: bash deploy.sh
# Requires: ansible installed on the server (apt install ansible)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_CONFIG="${APP_CONFIG:-bct-agent}"

echo "Deploying with config: $APP_CONFIG"

ansible-playbook \
  -e "app_config=$APP_CONFIG" \
  "$SCRIPT_DIR/playbooks/deploy.yml"
