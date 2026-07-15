#!/usr/bin/env sh

cd $(dirname $0)
PATH="$(pwd):$PATH" uv run "${1:-launcher.py}"

