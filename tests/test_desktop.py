from __future__ import annotations

import time
from pathlib import Path

import pytest

from transcript_anonymizer.desktop import (
    DesktopController,
    WindowedEta,
    eta_range,
    eta_seconds,
    format_elapsed,
    format_eta,
)


def _wait(controller: DesktopController) -> list[dict]:
    controller.join(2)
    assert not controller.busy
    return list(controller.messages())


class FakeDetector:
    def __init__(self, progress, *, slow=False, stage_progress=None):
        self.progress = progress
        self.slow = slow
        self.stage_progress = stage_progress

    def detect(self, segments):
        for index in range(1, len(segments) + 1):
            self.progress(index, len(segments))
            if self.slow:
                time.sleep(0.01)
        if self.stage_progress is not None:
            self.stage_progress("creating_review_files", len(segments), len(segments))
        return []


class FakeWorkflow:
    def __init__(self, *, signed=False, segment_count=2):
        self.signed = signed
        self.segment_count = segment_count
        self.prepared = []
        self.exported = []

    def prepare(self, source, run, policy, detector):
        self.prepared.append((source, run, policy, detector))
        detector.detect([{"id": f"s{i}", "text": "note"} for i in range(self.segment_count)])
        return self.status(run)

    def status(self, run):
        return {
            "run_id": "run-1",
            "run": str(run),
            "candidate": str(run / "revisions" / "r000001-x" / "candidate.docx"),
            "workbook": str(run / "revisions" / "r000001-x" / "review.xlsx"),
            "signed_off": self.signed,
        }

    def _load(self, run):
        return {"policy": {"schema_version": 1}}, None

    def import_review(self, run, workbook, detector):
        return self.status(run)

    def export(self, run, destination):
        self.exported.append((run, destination))
        return {"export_directory": str(destination), "signed_off": True}


def test_eta_is_conservative_until_throughput_exists():
    assert eta_seconds([], 1, 10) is None
    assert eta_seconds([(10.0, 1), (12.0, 3)], 3, 10) is None
    assert eta_seconds([(2.0 * i, i) for i in range(6)], 5, 10) == 11.0
    assert format_eta(None) == "Estimating..."
    assert format_eta(65) == "About 2 min remaining"


def test_deferred_detector_announces_loaded_before_first_segment(monkeypatch):
    from transcript_anonymizer.desktop import _DeferredDetector
    events = []
    class Inner:
        def _ensure_model(self):
            events.append('loaded')
        def detect(self, segments):
            assert events[-1] == 'processing'
            return []
    detector = _DeferredDetector({}, None, lambda stage, *_: events.append(stage))
    monkeypatch.setattr(detector, '_get', lambda: Inner())
    assert detector.detect([{'id': 's1', 'text': 'Synthetic'}]) == []
    assert events == ['model_loading', 'loaded', 'processing', 'creating_review_files']


def test_eta_range_smooths_timings_and_widens_with_variability():
    stable = eta_range([(10.0 * i, i) for i in range(9)], 8, 100)
    varied = eta_range([(t, i) for i, t in enumerate([0, 2, 20, 22, 40, 42, 60, 62, 80])], 8, 100)
    assert stable == pytest.approx((920, 1104))
    assert varied[1] - varied[0] > stable[1] - stable[0]
    spike = eta_range([(10.0 * i, i) for i in range(8)] + [(100.0, 8)], 8, 100)
    assert sum(stable) / 2 < sum(spike) / 2 < 30 * 92


def test_variable_timings_do_not_collapse_lower_bound():
    bounds = eta_range([(0, 0), (1, 1), (2, 2), (3, 3), (43, 4)], 4, 100)
    assert bounds[0] >= 43 / 4 * 96


def test_eta_waits_ten_seconds_and_holds_between_updates(monkeypatch):
    controller = DesktopController()
    now = [0.0]
    monkeypatch.setattr('transcript_anonymizer.desktop.time.monotonic', lambda: now[0])
    seen = []
    for index, timestamp in enumerate([0, 2, 4, 6, 8, 10, 11, 12, 19, 20]):
        now[0] = timestamp
        controller._progress('processing', index, 100)
        seen.append(list(controller.messages())[-1])
    assert all(event['eta'] is None for event in seen[:5])
    assert seen[5]['eta'] is not None
    assert all(event['eta'] == seen[5]['eta'] for event in seen[6:9])
    assert seen[9]['eta'] != seen[5]['eta']
    assert [event['completed'] for event in seen] == list(range(10))


def test_eta_recent_window_forgets_old_speed_and_rejects_incoherent_samples():
    samples = [(1000 + 10.0 * i, i + 1) for i in range(7)]
    assert eta_range([(0, 0), *samples], 7, 100) == eta_range(samples, 7, 100)
    for bad in (
        [(0, 0), (2, 1), (4, 2), (6, 1), (8, 4)],
        [(0, 0), (2, 1), (4, 2), (3, 3), (8, 4)],
        [(0, 0), (2, 1), (4, 2), (6, 2), (8, 4)],
    ):
        assert eta_range(bad, 4, 100) is None
    assert eta_range([(i, i) for i in range(5)], 4, 100) is None
    assert eta_range(samples, 6, 100) is None
    assert eta_range(samples, 7, 7) is None


def test_window_pools_progress_instead_of_picking_fastest_or_narrowest_sample():
    results = []
    for counts in ([0, 1, 2, 3, 20], [0, 8, 12, 16, 20]):
        estimate = WindowedEta()
        for now, completed in zip([0, 1, 2, 3, 10], counts):
            result = estimate.update(now, completed, 100)
        results.append(result)
    assert results[0] == results[1] == (40, 48)


def test_fast_burst_cannot_undercut_stage_average_and_completion_clears_eta():
    estimate = WindowedEta()
    for now, completed in [(0, 0), (3, 1), (6, 2), (9, 3), (12, 4)]:
        estimate.update(now, completed, 200)
    bounds = estimate.update(22, 104, 200)
    assert bounds[0] >= 22 / 104 * 96
    assert estimate.update(23, 104, 200) == bounds
    assert estimate.update(24, 200, 200) is None


@pytest.mark.parametrize(('seconds', 'expected'), [
    (None, 'Estimating...'),
    ((180, 300), 'About 3-5 min remaining'),
    ((181, 299), 'About 4-5 min remaining'),
    ((2, 59), 'Less than 1 min remaining'),
    ((0, 121), 'Up to about 3 min remaining'),
    ((60, 60), 'About 1 min remaining'),
    (float('inf'), 'Estimating...'),
])
def test_eta_display_avoids_false_second_precision(seconds, expected):
    assert format_eta(seconds) == expected


@pytest.mark.parametrize(('seconds', 'expected'), [
    (0, 'Elapsed 0:00:00'), (65.9, 'Elapsed 0:01:05'),
    (3661, 'Elapsed 1:01:01'), (360000, 'Elapsed 100:00:00'),
])
def test_elapsed_uses_hours_minutes_seconds(seconds, expected):
    assert format_elapsed(seconds) == expected


def test_progress_resets_eta_after_stage_total_or_count_change(monkeypatch):
    controller = DesktopController()
    ticks = iter(range(0, 1000, 10))
    monkeypatch.setattr('transcript_anonymizer.desktop.time.monotonic', lambda: next(ticks))
    for i in range(5):
        controller._progress('processing', i, 100)
    estimate = list(controller.messages())[-1]['eta']
    assert estimate is not None
    controller._progress('processing', 4, 100)
    assert list(controller.messages())[-1]['eta'] == estimate
    for stage, count, total in [('processing', 5, 101), ('other', 6, 101), ('other', 1, 101)]:
        controller._progress(stage, count, total)
        assert list(controller.messages())[-1]['eta'] is None


def test_controller_processes_in_one_worker_and_reports_stages(tmp_path: Path):
    fake = FakeWorkflow()
    made = []

    def factory(policy, model_path, progress):
        made.append((policy, model_path))
        return FakeDetector(
            lambda completed, total: progress("processing", completed, total),
            stage_progress=progress,
        )

    controller = DesktopController(
        model_path=tmp_path / "model",
        workflow_module=fake,
        detector_factory=factory,
    )
    controller.start_process(tmp_path / "source.docx", tmp_path / "workspace")
    events = _wait(controller)
    assert fake.prepared
    assert fake.prepared[0][1].parent == tmp_path / "workspace"
    assert not fake.prepared[0][1].exists()
    assert made[0][0]["schema_version"] == 1
    assert made[0][1] == tmp_path / "model"
    assert [event["stage"] for event in events if event["kind"] == "progress"] == [
        "processing",
        "processing",
        "creating_review_files",
    ]
    assert any(event["kind"] == "result" for event in events)
    assert any(event["kind"] == "finished" for event in events)


def test_controller_cancellation_waits_for_worker_and_has_no_result(tmp_path: Path):
    fake = FakeWorkflow(segment_count=100)

    def factory(policy, model_path, progress):
        return FakeDetector(
            lambda completed, total: progress("processing", completed, total), slow=True
        )

    controller = DesktopController(workflow_module=fake, detector_factory=factory)
    controller.start_process(tmp_path / "source.docx", tmp_path / "workspace")
    time.sleep(0.015)
    assert controller.cancel()
    events = _wait(controller)
    kinds = [event["kind"] for event in events]
    assert "cancel_requested" in kinds
    assert "cancelled" in kinds
    assert "result" not in kinds
    assert kinds[-1] == "finished"


def test_controller_refuses_unsigned_export(tmp_path: Path):
    fake = FakeWorkflow(signed=False)
    controller = DesktopController(workflow_module=fake, detector_factory=lambda *_: None)
    controller.select_run(tmp_path / "run")
    _wait(controller)
    controller.export(tmp_path / "workspace")
    events = _wait(controller)
    assert any(event["kind"] == "error" and "sign-off" in event["message"] for event in events)
    assert fake.exported == []


def test_controller_exports_signed_run_to_unique_child(tmp_path: Path):
    fake = FakeWorkflow(signed=True)
    controller = DesktopController(workflow_module=fake, detector_factory=lambda *_: None)
    controller.select_run(tmp_path / "run")
    _wait(controller)
    controller.export(tmp_path / "workspace")
    events = _wait(controller)
    assert fake.exported
    assert fake.exported[0][1].name.startswith("export-")
    assert controller.status["signed_off"]
    assert any(event["kind"] == "result" for event in events)
