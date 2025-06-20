#!/bin/bash
# A script to set up a basic development environment.

# ---
# Exit immediately if a command exits with a non-zero status.
# This prevents errors from being ignored.
# ---
set -e

# ---
# Provide feedback to the user
# ---
echo "🚀 Starting development environment setup..."

# ---
# 1. Install 'uv' - A super-fast Python package installer
# Note: We are trusting the source here for convenience.
# ---
echo "📦 Installing uv from astral.sh..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# ---
# 2. Update package lists and install essential tools via apt
# - 'apt update' refreshes the list of available packages.
# - 'apt install -y' installs packages and automatically says "yes" to prompts.
# ---
echo "🔧 Updating package lists and installing tools (micro, neovim)..."
sudo apt update
sudo apt install -y micro neovim git btop ncdu lsd python3-dev # Added git to be explicit

# ---
# 3. Set up LazyVim configuration for Neovim
# - We check if the config directory already exists to avoid errors on re-runs.
# ---
NVIM_CONFIG_DIR="$HOME/.config/nvim"
if [ ! -d "$NVIM_CONFIG_DIR" ]; then
    echo " cloning LazyVim starter configuration..."
    git clone https://github.com/LazyVim/starter "$NVIM_CONFIG_DIR"
else
    echo "✔️ Neovim config already exists, skipping clone."
fi

# ---
# All done!
# ---
echo "✅ Setup complete! Enjoy your new tools."

