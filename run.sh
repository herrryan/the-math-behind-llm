#!/usr/bin/env bash
# ==============================================================================
# the-math-behind-llm — Dynamic Port Startup Runner
# ==============================================================================
set -e

BOLD="\033[1m"
PURPLE="\033[38;5;141m"
GREEN="\033[38;5;120m"
SLATE="\033[38;5;244m"
RESET="\033[0m"

export PORT="${PORT:-8000}"

echo -e "${PURPLE}✦${RESET} ${BOLD}Starting the-math-behind-llm on port ${PORT}...${RESET}"
cd "$(dirname "$0")"

# Activate virtual environment if available
if [ -d ".venv" ]; then
    echo -e "  ${SLATE}↳ Activating .venv...${RESET}"
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    echo -e "  ${SLATE}↳ Activating parent .venv...${RESET}"
    source ../.venv/bin/activate
elif [ -d "venv" ]; then
    echo -e "  ${SLATE}↳ Activating venv...${RESET}"
    source venv/bin/activate
fi

# Check requirements
if [ -f "requirements.txt" ] && [ -d ".venv" ]; then
    pip install -r requirements.txt --quiet || true
fi

echo -e "${GREEN}✔${RESET} ${BOLD}Executing:${RESET} PORT="${PORT:-8000}" python3 build.py"
eval exec PORT="${PORT:-8000}" python3 build.py
