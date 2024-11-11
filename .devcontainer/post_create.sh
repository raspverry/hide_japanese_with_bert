#!/usr/bin/env bash
# .devcontainer/post_create.sh

set -euo pipefail

# Define workspace folder path
containerWorkspaceFolder="/workspace"

# Set ownership of /workspace
sudo chown -R vscode:vscode "$containerWorkspaceFolder"
echo "Set permissions for $containerWorkspaceFolder to vscode:vscode"

# Add /workspace to Git safe directory
sudo git config --system --add safe.directory "$containerWorkspaceFolder"
echo "Added $containerWorkspaceFolder to git safe directory"

# Additional setup steps can be added here
