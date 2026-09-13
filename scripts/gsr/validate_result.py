from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.gsr import GSRContractError, GSRFrameStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valide un résultat football-tracking.gsr/v1 avant import.",
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("--home-cluster", choices=["A", "B"], default="B")
    args = parser.parse_args()
    try:
        store = GSRFrameStore.load(
            args.result,
            home_team_cluster=args.home_cluster,
        )
    except GSRContractError as exc:
        print(f"INVALIDE: {exc}")
        return 1

    roles = Counter(obj.role for frame in store.frames for obj in frame.objects)
    unique_tracks = {obj.track_id for frame in store.frames for obj in frame.objects}
    print(f"VALIDE: {store.engine} revision={store.engine_revision}")
    print(
        f"frames={len(store.frames)} tracks={len(unique_tracks)} "
        f"intervalle={store.timestamps[0]}..{store.timestamps[-1]} ms"
    )
    print("roles=" + ", ".join(f"{key}:{value}" for key, value in sorted(roles.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
