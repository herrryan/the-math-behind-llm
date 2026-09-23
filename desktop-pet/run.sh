#!/usr/bin/env bash
# One-click launcher for AI Desktop Pet (Clippy 2.0)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " Starting AI Desktop Pet (Clippy 2.0)..."
echo " Controls:"
echo " - Left Drag: Move pet anywhere on screen"
echo " - Double Click: Ask pet for current status"
echo " - Right Click: Switch persona (Clippy/Cat/Hacker/Zen)"
echo " - Speech Bubble: Type in the input box to chat"
echo "=================================================="

uv run --with pyqt6 python3 desktop_pet.py
