#!/usr/bin/env bash
set -euo pipefail

# Official sn-gamestate currently requires Python 3.9 and TrackLab 1.3.24.
# Keep it outside Django's Python 3.12 environment.
SN_GSR_ROOT="${1:-${HOME}/.football-tracking/gsr/sn-gamestate}"
SN_GSR_REVISION="${SN_GSR_REVISION:-1c958345067218297d221e45e1a6405f975f83e0}"

need_apt=0
for command_name in git curl ffmpeg; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    need_apt=1
  fi
done
if [ "${need_apt}" -eq 1 ]; then
  sudo apt-get update
  sudo apt-get install -y git curl ffmpeg build-essential
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

mkdir -p "$(dirname "${SN_GSR_ROOT}")"
if [ ! -d "${SN_GSR_ROOT}/.git" ]; then
  git clone https://github.com/SoccerNet/sn-gamestate.git "${SN_GSR_ROOT}"
elif [ -n "$(git -C "${SN_GSR_ROOT}" status --porcelain)" ]; then
  echo "Le dépôt externe sn-gamestate contient des changements locaux." >&2
  echo "Ils sont conservés. Nettoyez-les ou choisissez un autre dossier." >&2
  exit 1
else
  git -C "${SN_GSR_ROOT}" fetch origin
fi

git -C "${SN_GSR_ROOT}" checkout --detach "${SN_GSR_REVISION}"
cd "${SN_GSR_ROOT}"
uv venv --python 3.9
uv pip install -e .
uv run mim install mmcv==2.0.1

echo "sn-gamestate installé séparément dans ${SN_GSR_ROOT}"
echo "Révision figée : $(git -C "${SN_GSR_ROOT}" rev-parse HEAD)"
