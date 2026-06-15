#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

python prepare_paper_figures.py "$@"
