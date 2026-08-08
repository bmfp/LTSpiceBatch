#!/usr/bin/env sh

cd $(dirname $0)
PATH="$(pwd):$PATH" uv --managed-python run "${1:-diapo.py}" & < /dev/null
