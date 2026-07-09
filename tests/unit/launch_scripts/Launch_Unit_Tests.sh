#!/usr/bin/env bash
set -xeuo pipefail

python -m coverage run --data-file="${COVERAGE_FILE:-.coverage}" --source=molt -m pytest tests/unit
