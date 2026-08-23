#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel setuptools
.venv/bin/python -m pip install -r requirements.txt
if [[ "${1:-}" == "--with-ml" ]]; then
  .venv/bin/python -m pip install -r requirements-ml.txt
fi
[[ -f .env ]] || cp .env.example .env
.venv/bin/python manage.py migrate
.venv/bin/python manage.py check
.venv/bin/python manage.py diagnose

echo "Installation terminée. Lancez le serveur et le worker dans deux terminaux."
