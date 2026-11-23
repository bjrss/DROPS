#!/bin/bash
# Waits for the final plates, then builds every deliverable in order.
cd "$(dirname "$0")/.."
until grep -q ALL_PLATES_DONE build/plates_master.log 2>/dev/null; do sleep 30; done
echo "PLATES_DONE $(date +%T)"
set -e
python3 tools/deliver.py compose 9x16 4x5 16x9
echo "COMPOSE_DONE $(date +%T)"
python3 tools/deliver.py main
python3 tools/deliver.py srt
python3 tools/deliver.py capsules 9x16 4x5 16x9
echo "CAPSULES_DONE $(date +%T)"
python3 tools/deliver.py stems
python3 tools/make_brief.py
echo "FINALIZE_DONE $(date +%T)"
