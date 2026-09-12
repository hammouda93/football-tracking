from __future__ import annotations

from dataclasses import dataclass

from .types import FrameSignal, TimeSpan


@dataclass(slots=True)
class PeriodDetectionResult:
    periods: list[TimeSpan]
    active_blocks: list[TimeSpan]
    confidence: float
    requires_review: bool
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "periods": [item.to_dict() for item in self.periods],
            "active_blocks": [item.to_dict() for item in self.active_blocks],
            "confidence": self.confidence,
            "requires_review": self.requires_review,
            "diagnostics": self.diagnostics,
        }


class PeriodDetector:
    """Suggest half boundaries from long, field-dominant video regions.

    The suggestions are intentionally reviewable. A match clock/OCR provider can later
    contribute stronger anchors without changing this interface.
    """

    def __init__(
        self,
        field_threshold: float = 0.24,
        bridge_gap_ms: int = 75_000,
        minimum_block_ms: int = 12 * 60_000,
    ):
        self.field_threshold = field_threshold
        self.bridge_gap_ms = bridge_gap_ms
        self.minimum_block_ms = minimum_block_ms

    def detect(self, signals: list[FrameSignal], duration_ms: int) -> PeriodDetectionResult:
        if not signals:
            return self._fallback(duration_ms, "no_signals")
        sample_interval = self._sample_interval(signals)
        raw_blocks: list[TimeSpan] = []
        current_start: int | None = None
        last_active: int | None = None
        values: list[float] = []
        block_scores: list[float] = []

        for signal in signals:
            is_field = signal.field_score >= self.field_threshold
            if is_field:
                if current_start is None:
                    current_start = signal.timestamp_ms
                    values = []
                last_active = signal.timestamp_ms
                values.append(signal.field_score)
            elif current_start is not None and last_active is not None:
                if signal.timestamp_ms - last_active <= self.bridge_gap_ms:
                    continue
                end_ms = last_active + sample_interval
                score = sum(values) / max(len(values), 1)
                if end_ms - current_start >= self.minimum_block_ms:
                    raw_blocks.append(TimeSpan(current_start, end_ms, score, "field_block"))
                    block_scores.append(score)
                current_start = None
                last_active = None
                values = []

        if current_start is not None and last_active is not None:
            end_ms = min(duration_ms, last_active + sample_interval)
            score = sum(values) / max(len(values), 1)
            if end_ms - current_start >= self.minimum_block_ms:
                raw_blocks.append(TimeSpan(current_start, end_ms, score, "field_block"))
                block_scores.append(score)

        central_break = self._select_central_break(
            signals,
            duration_ms,
            sample_interval,
        )
        periods = (
            central_break[0]
            if central_break is not None
            else self._select_two_halves(raw_blocks, duration_ms)
        )
        if len(periods) != 2:
            fallback = self._fallback(duration_ms, "unable_to_isolate_two_halves")
            fallback.active_blocks = raw_blocks
            return fallback

        confidence = min(period.confidence for period in periods)
        separation = periods[1].start_ms - periods[0].end_ms
        if central_break is not None:
            confidence *= min(1.0, max(0.65, separation / 120_000))
        else:
            confidence *= min(1.0, max(0.35, separation / (8 * 60_000)))
        diagnostics = {
            "sample_interval_ms": sample_interval,
            "field_threshold": self.field_threshold,
            "halftime_gap_ms": separation,
            "block_count": len(raw_blocks),
            "detection_method": (
                "central_broadcast_break"
                if central_break is not None
                else "field_blocks"
            ),
        }
        if central_break is not None:
            diagnostics.update(central_break[1])
        return PeriodDetectionResult(
            periods=periods,
            active_blocks=raw_blocks,
            confidence=round(confidence, 4),
            requires_review=confidence < 0.86,
            diagnostics=diagnostics,
        )

    def _select_central_break(
        self,
        signals: list[FrameSignal],
        duration_ms: int,
        sample_interval: int,
    ) -> tuple[list[TimeSpan], dict] | None:
        """Find a short edited half-time break around the middle of a broadcast.

        Some source videos remove almost the entire half-time interval. Their break
        lasts less than the normal field-block bridge, so it must be detected from
        consecutive non-field samples near the centre of the recording.
        """

        if len(signals) < 8 or duration_ms <= 0:
            return None

        search_start = int(duration_ms * 0.35)
        search_end = int(duration_ms * 0.65)
        inactive_runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for index, signal in enumerate(signals):
            is_candidate = (
                search_start <= signal.timestamp_ms <= search_end
                and signal.field_score < self.field_threshold
            )
            if is_candidate and run_start is None:
                run_start = index
            elif not is_candidate and run_start is not None:
                inactive_runs.append((run_start, index - 1))
                run_start = None
        if run_start is not None:
            inactive_runs.append((run_start, len(signals) - 1))

        candidates: list[tuple[float, int, int, int, int, float, float]] = []
        minimum_gap = max(20_000, int(sample_interval * 1.8))
        midpoint = duration_ms / 2.0
        search_radius = max(1.0, duration_ms * 0.15)
        for start_index, end_index in inactive_runs:
            before_index = start_index - 1
            after_index = end_index + 1
            if before_index < 0 or after_index >= len(signals):
                continue
            before = signals[before_index]
            after = signals[after_index]
            if (
                before.field_score < self.field_threshold
                or after.field_score < self.field_threshold
            ):
                continue
            separation = after.timestamp_ms - before.timestamp_ms
            if separation < minimum_gap or separation > 20 * 60_000:
                continue

            support_window = 4 * 60_000
            before_support = [
                signal
                for signal in signals
                if before.timestamp_ms - support_window
                <= signal.timestamp_ms
                <= before.timestamp_ms
            ]
            after_support = [
                signal
                for signal in signals
                if after.timestamp_ms
                <= signal.timestamp_ms
                <= after.timestamp_ms + support_window
            ]
            before_ratio = self._field_ratio(before_support)
            after_ratio = self._field_ratio(after_support)
            if min(before_ratio, after_ratio) < 0.45:
                continue

            break_midpoint = (before.timestamp_ms + after.timestamp_ms) / 2.0
            centrality = max(0.0, 1.0 - abs(break_midpoint - midpoint) / search_radius)
            duration_score = min(1.0, separation / 120_000)
            support_score = (before_ratio + after_ratio) / 2.0
            score = centrality * 0.55 + duration_score * 0.25 + support_score * 0.20
            candidates.append(
                (
                    score,
                    before_index,
                    after_index,
                    before.timestamp_ms,
                    after.timestamp_ms,
                    before_ratio,
                    after_ratio,
                )
            )

        if not candidates:
            return None

        (
            score,
            before_index,
            after_index,
            first_end,
            second_start,
            before_ratio,
            after_ratio,
        ) = max(candidates, key=lambda item: item[0])
        first_field = next(
            (
                signal
                for signal in signals[: before_index + 1]
                if signal.field_score >= self.field_threshold
            ),
            None,
        )
        second_field_signals = [
            signal
            for signal in signals[after_index:]
            if signal.field_score >= self.field_threshold
        ]
        if first_field is None or not second_field_signals:
            return None
        second_end = min(
            duration_ms,
            second_field_signals[-1].timestamp_ms + sample_interval,
        )
        if first_end - first_field.timestamp_ms < 30 * 60_000:
            return None
        if second_end - second_start < 30 * 60_000:
            return None

        period_confidence = min(0.85, 0.55 + score * 0.30)
        periods = [
            TimeSpan(
                first_field.timestamp_ms,
                first_end,
                period_confidence,
                "1re mi-temps",
            ),
            TimeSpan(
                second_start,
                second_end,
                period_confidence,
                "2e mi-temps",
            ),
        ]
        return periods, {
            "central_break_score": round(score, 4),
            "field_ratio_before_break": round(before_ratio, 4),
            "field_ratio_after_break": round(after_ratio, 4),
        }

    def _field_ratio(self, signals: list[FrameSignal]) -> float:
        if not signals:
            return 0.0
        return sum(
            signal.field_score >= self.field_threshold for signal in signals
        ) / len(signals)

    def _select_two_halves(self, blocks: list[TimeSpan], duration_ms: int) -> list[TimeSpan]:
        if len(blocks) >= 2:
            candidates = sorted(blocks, key=lambda item: item.duration_ms, reverse=True)[:4]
            best_pair = None
            best_score = -1.0
            for first in candidates:
                for second in candidates:
                    if first.start_ms >= second.start_ms:
                        continue
                    separation = second.start_ms - first.end_ms
                    if separation < 60_000:
                        continue
                    duration_balance = min(first.duration_ms, second.duration_ms) / max(
                        first.duration_ms, second.duration_ms
                    )
                    score = duration_balance + first.confidence + second.confidence
                    if score > best_score:
                        best_score = score
                        best_pair = (first, second)
            if best_pair:
                first, second = best_pair
                return [
                    TimeSpan(first.start_ms, first.end_ms, min(0.95, first.confidence), "1re mi-temps"),
                    TimeSpan(second.start_ms, second.end_ms, min(0.95, second.confidence), "2e mi-temps"),
                ]
        if len(blocks) == 1 and blocks[0].duration_ms >= 70 * 60_000:
            block = blocks[0]
            midpoint = block.start_ms + block.duration_ms // 2
            return [
                TimeSpan(block.start_ms, midpoint - 5 * 60_000, 0.45, "1re mi-temps"),
                TimeSpan(midpoint + 5 * 60_000, block.end_ms, 0.45, "2e mi-temps"),
            ]
        return []

    def _fallback(self, duration_ms: int, reason: str) -> PeriodDetectionResult:
        preamble = min(max(int(duration_ms * 0.03), 0), 10 * 60_000)
        usable_end = max(preamble, duration_ms - min(int(duration_ms * 0.02), 5 * 60_000))
        halftime_gap = min(15 * 60_000, int(duration_ms * 0.10))
        midpoint = preamble + (usable_end - preamble) // 2
        periods = [
            TimeSpan(preamble, max(preamble, midpoint - halftime_gap // 2), 0.2, "1re mi-temps"),
            TimeSpan(min(usable_end, midpoint + halftime_gap // 2), usable_end, 0.2, "2e mi-temps"),
        ]
        return PeriodDetectionResult(
            periods=periods,
            active_blocks=[],
            confidence=0.2,
            requires_review=True,
            diagnostics={"fallback_reason": reason},
        )

    @staticmethod
    def _sample_interval(signals: list[FrameSignal]) -> int:
        if len(signals) < 2:
            return 1000
        deltas = [
            current.timestamp_ms - previous.timestamp_ms
            for previous, current in zip(signals, signals[1:])
            if current.timestamp_ms > previous.timestamp_ms
        ]
        return int(sum(deltas) / len(deltas)) if deltas else 1000
