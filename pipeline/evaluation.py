from __future__ import annotations

import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PERSON_ROLES = {"player", "goalkeeper", "referee"}


@dataclass(frozen=True, slots=True)
class TruthObject:
    timestamp_ms: int
    object_id: str
    role: str
    bbox: tuple[float, float, float, float]
    team_key: str | None = None
    shirt_number: int | None = None
    pitch_x: float | None = None
    pitch_y: float | None = None


def _optional_float(value) -> float | None:
    if value in {None, ""}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _optional_int(value) -> int | None:
    if value in {None, ""}:
        return None
    parsed = int(value)
    return parsed if 0 <= parsed <= 99 else None


def _team(value) -> str | None:
    normalized = str(value or "").strip().lower()
    return {
        "home": "home",
        "t1": "home",
        "1": "home",
        "away": "away",
        "t2": "away",
        "2": "away",
    }.get(normalized)


def load_ground_truth(path: str | Path) -> dict[int, list[TruthObject]]:
    grouped: defaultdict[int, list[TruthObject]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"timestamp_ms", "object_id", "role", "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Colonnes vérité terrain manquantes: " + ", ".join(sorted(missing))
            )
        for line, row in enumerate(reader, start=2):
            if "reviewed" in (reader.fieldnames or []):
                reviewed = str(row.get("reviewed") or "").strip().lower()
                if reviewed not in {"1", "yes", "oui", "true"}:
                    raise ValueError(
                        f"Ligne {line}: corrige la pré-annotation puis mets reviewed=YES."
                    )
            timestamp_ms = int(row["timestamp_ms"])
            role = str(row["role"]).strip().lower()
            if timestamp_ms < 0 or role not in PERSON_ROLES:
                raise ValueError(f"Ligne {line}: timestamp ou rôle invalide.")
            bbox = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            if not all(math.isfinite(value) for value in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"Ligne {line}: boîte invalide.")
            object_id = str(row["object_id"]).strip()
            if not object_id:
                raise ValueError(f"Ligne {line}: object_id vide.")
            grouped[timestamp_ms].append(
                TruthObject(
                    timestamp_ms=timestamp_ms,
                    object_id=object_id,
                    role=role,
                    bbox=bbox,
                    team_key=_team(row.get("team")),
                    shirt_number=_optional_int(row.get("shirt_number")),
                    pitch_x=_optional_float(row.get("pitch_x")),
                    pitch_y=_optional_float(row.get("pitch_y")),
                )
            )
    if not grouped:
        raise ValueError("Le fichier vérité terrain ne contient aucune annotation.")
    return dict(grouped)


def validate_ground_truth(path: str | Path) -> dict:
    frames = load_ground_truth(path)
    rows = [item for objects in frames.values() for item in objects]
    return {
        "frames": len(frames),
        "objects": len(rows),
        "start_ms": min(frames),
        "end_ms": max(frames),
        "team_labels": sum(item.team_key is not None for item in rows),
        "jersey_labels": sum(item.shirt_number is not None for item in rows),
        "pitch_labels": sum(
            item.pitch_x is not None and item.pitch_y is not None for item in rows
        ),
    }


def _iou(first, second) -> float:
    try:
        if len(first) != 4 or len(second) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _maximum_weight_assignment(weights: list[list[int]]) -> int:
    """Return the exact maximum one-to-one assignment weight.

    This is the rectangular Hungarian algorithm. Keeping it local avoids making
    SciPy a mandatory dependency just to score a short validation sample.
    """

    if not weights or not weights[0]:
        return 0
    row_count = len(weights)
    column_count = len(weights[0])
    transposed = row_count > column_count
    matrix = (
        [list(row) for row in zip(*weights, strict=True)]
        if transposed
        else [list(row) for row in weights]
    )
    rows, columns = len(matrix), len(matrix[0])
    maximum = max(max(row, default=0) for row in matrix)
    # The canonical algorithm minimizes cost. max-weight becomes max-w.
    costs = [[maximum - int(value) for value in row] for row in matrix]
    u = [0] * (rows + 1)
    v = [0] * (columns + 1)
    matched_row = [0] * (columns + 1)
    previous_column = [0] * (columns + 1)
    for row in range(1, rows + 1):
        matched_row[0] = row
        column = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate - 1]
                    - u[current_row]
                    - v[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            next_column = previous_column[column]
            matched_row[column] = matched_row[next_column]
            column = next_column
            if column == 0:
                break
    return sum(
        matrix[row - 1][column - 1]
        for column, row in enumerate(matched_row[1:], start=1)
        if row
    )


def _same_role(first: str, second: str) -> bool:
    if first == second:
        return True
    return {first, second} <= {"player", "goalkeeper"}


def _nearest_timestamp(timestamps: list[int], target: int, tolerance_ms: int) -> int | None:
    position = bisect.bisect_left(timestamps, target)
    candidates = timestamps[max(0, position - 1) : position + 1]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - target))
    return nearest if abs(nearest - target) <= tolerance_ms else None


def evaluate_tracking(
    tracking_path: str | Path,
    truth_path: str | Path,
    *,
    timestamp_tolerance_ms: int = 80,
    iou_threshold: float = 0.50,
) -> dict:
    truth = load_ground_truth(truth_path)
    truth_timestamps = sorted(truth)
    used_truth_frames: set[int] = set()
    counts: Counter = Counter()
    predicted_ids_by_truth: defaultdict[str, Counter[str]] = defaultdict(Counter)
    truth_detection_counts: Counter[str] = Counter()
    predicted_detection_counts: Counter[str] = Counter()
    last_predicted_by_truth: dict[str, str] = {}
    pitch_errors: list[float] = []

    with Path(tracking_path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            frame = payload.get("frame") or {}
            timestamp_ms = int(frame.get("timestamp_ms", -1))
            nearest = _nearest_timestamp(
                truth_timestamps,
                timestamp_ms,
                max(1, int(timestamp_tolerance_ms)),
            )
            if nearest is None or nearest in used_truth_frames:
                continue
            used_truth_frames.add(nearest)
            counts["evaluated_frames"] += 1
            expected = truth[nearest]
            predicted = [
                obj
                for obj in frame.get("objects", [])
                if str(obj.get("role", "")) in PERSON_ROLES
            ]
            truth_detection_counts.update(item.object_id for item in expected)
            predicted_detection_counts.update(
                str(item.get("track_id", "")) for item in predicted
            )
            candidates: list[tuple[float, int, int]] = []
            for expected_index, truth_obj in enumerate(expected):
                for predicted_index, predicted_obj in enumerate(predicted):
                    if not _same_role(truth_obj.role, str(predicted_obj.get("role", ""))):
                        continue
                    score = _iou(truth_obj.bbox, predicted_obj.get("bbox_xyxy") or [])
                    if score >= iou_threshold:
                        candidates.append((score, expected_index, predicted_index))
            matched_truth: set[int] = set()
            matched_predictions: set[int] = set()
            for _score, expected_index, predicted_index in sorted(candidates, reverse=True):
                if expected_index in matched_truth or predicted_index in matched_predictions:
                    continue
                matched_truth.add(expected_index)
                matched_predictions.add(predicted_index)
                counts["tp"] += 1
                truth_obj = expected[expected_index]
                predicted_obj = predicted[predicted_index]
                predicted_id = str(predicted_obj.get("track_id", ""))
                predicted_ids_by_truth[truth_obj.object_id][predicted_id] += 1
                previous = last_predicted_by_truth.get(truth_obj.object_id)
                if previous is not None and previous != predicted_id:
                    counts["id_switches"] += 1
                last_predicted_by_truth[truth_obj.object_id] = predicted_id
                if truth_obj.team_key is not None:
                    counts["team_labeled"] += 1
                    counts["team_correct"] += int(
                        predicted_obj.get("team_key") == truth_obj.team_key
                    )
                if truth_obj.shirt_number is not None:
                    counts["jersey_labeled"] += 1
                    predicted_number = predicted_obj.get("shirt_number")
                    counts["jersey_covered"] += int(predicted_number is not None)
                    counts["jersey_correct"] += int(
                        predicted_number == truth_obj.shirt_number
                    )
                if truth_obj.pitch_x is not None and truth_obj.pitch_y is not None:
                    counts["pitch_labeled"] += 1
                    pitch_x = predicted_obj.get("pitch_x")
                    pitch_y = predicted_obj.get("pitch_y")
                    if pitch_x is not None and pitch_y is not None:
                        counts["pitch_covered"] += 1
                        pitch_errors.append(
                            math.hypot(
                                float(pitch_x) - truth_obj.pitch_x,
                                float(pitch_y) - truth_obj.pitch_y,
                            )
                        )
            counts["fn"] += len(expected) - len(matched_truth)
            counts["fp"] += len(predicted) - len(matched_predictions)

    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    consistent = sum(max(votes.values()) for votes in predicted_ids_by_truth.values() if votes)
    identity_total = sum(sum(votes.values()) for votes in predicted_ids_by_truth.values())
    fragments = [len(votes) for votes in predicted_ids_by_truth.values() if votes]
    team_accuracy = counts["team_correct"] / max(counts["team_labeled"], 1)
    jersey_accuracy = counts["jersey_correct"] / max(counts["jersey_labeled"], 1)
    jersey_coverage = counts["jersey_covered"] / max(counts["jersey_labeled"], 1)
    pitch_coverage = counts["pitch_covered"] / max(counts["pitch_labeled"], 1)
    identity_consistency = consistent / max(identity_total, 1)
    truth_ids = sorted(truth_detection_counts)
    predicted_ids = sorted(predicted_detection_counts)
    identity_matrix = [
        [int(predicted_ids_by_truth[truth_id][predicted_id]) for predicted_id in predicted_ids]
        for truth_id in truth_ids
    ]
    identity_true_positives = _maximum_weight_assignment(identity_matrix)
    identity_false_negatives = sum(truth_detection_counts.values()) - identity_true_positives
    identity_false_positives = (
        sum(predicted_detection_counts.values()) - identity_true_positives
    )
    idf1 = (2 * identity_true_positives) / max(
        2 * identity_true_positives
        + identity_false_positives
        + identity_false_negatives,
        1,
    )
    detection_accuracy = counts["tp"] / max(
        counts["tp"] + counts["fp"] + counts["fn"],
        1,
    )
    association_sum = 0.0
    for truth_id, votes in predicted_ids_by_truth.items():
        for predicted_id, pair_count in votes.items():
            union = (
                truth_detection_counts[truth_id]
                + predicted_detection_counts[predicted_id]
                - pair_count
            )
            association_sum += pair_count * pair_count / max(union, 1)
    association_accuracy = association_sum / max(counts["tp"], 1)
    # HOTA at the configured IoU threshold. This is deliberately named HOTA@.50,
    # not the challenge's threshold-integrated HOTA/GS-HOTA score.
    hota_50 = math.sqrt(max(0.0, detection_accuracy * association_accuracy))
    issues: list[str] = []
    if counts["evaluated_frames"] < max(10, int(len(truth) * 0.80)):
        issues.append("Moins de 80 % des images annotées ont été comparées.")
    if precision < 0.80:
        issues.append("Précision détection inférieure à 80 %.")
    if recall < 0.80:
        issues.append("Rappel détection inférieur à 80 %.")
    if identity_consistency < 0.80:
        issues.append("Cohérence d’identité inférieure à 80 %.")
    if idf1 < 0.70:
        issues.append("IDF1 inférieur à 70 %.")
    if counts["team_labeled"] and team_accuracy < 0.85:
        issues.append("Exactitude équipe inférieure à 85 %.")
    if counts["jersey_labeled"] and (jersey_coverage < 0.30 or jersey_accuracy < 0.25):
        issues.append("Couverture ou exactitude des numéros insuffisante.")
    if pitch_errors and float(sum(pitch_errors) / len(pitch_errors)) > 5.0:
        issues.append("Erreur terrain moyenne supérieure à 5 mètres.")

    return {
        "status": "evaluated",
        "verdict": "pass" if not issues else "fail",
        "annotated_frames": len(truth),
        "evaluated_frames": int(counts["evaluated_frames"]),
        "true_positives": int(counts["tp"]),
        "false_positives": int(counts["fp"]),
        "false_negatives": int(counts["fn"]),
        "precision_pct": round(100 * precision, 2),
        "recall_pct": round(100 * recall, 2),
        "f1_pct": round(100 * f1, 2),
        "id_switches": int(counts["id_switches"]),
        "id_true_positives": int(identity_true_positives),
        "id_false_positives": int(identity_false_positives),
        "id_false_negatives": int(identity_false_negatives),
        "idf1_pct": round(100 * idf1, 2),
        "hota_50_pct": round(100 * hota_50, 2),
        "detection_accuracy_pct": round(100 * detection_accuracy, 2),
        "association_accuracy_pct": round(100 * association_accuracy, 2),
        "identity_consistency_pct": round(100 * identity_consistency, 2),
        "mean_fragments_per_player": round(sum(fragments) / max(len(fragments), 1), 2),
        "team_accuracy_pct": round(100 * team_accuracy, 2),
        "jersey_coverage_pct": round(100 * jersey_coverage, 2),
        "jersey_accuracy_pct": round(100 * jersey_accuracy, 2),
        "pitch_coverage_pct": round(100 * pitch_coverage, 2),
        "mean_pitch_error_m": (
            round(sum(pitch_errors) / len(pitch_errors), 3) if pitch_errors else None
        ),
        "issues": issues,
    }
