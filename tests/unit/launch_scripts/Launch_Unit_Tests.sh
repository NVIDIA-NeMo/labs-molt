#!/usr/bin/env bash
set -xeuo pipefail

python -m coverage run -m pytest tests/unit
