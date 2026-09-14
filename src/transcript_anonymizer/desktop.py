"""Small native desktop frontend for the offline review workflow.

The controller is intentionally independent of Tk.  It owns one cooperative
worker thread and communicates with the UI through a queue of plain dictionaries;
this keeps the workflow testable on headless build machines and prevents Tk
objects from crossing the worker boundary.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

from . import workflow
from .errors import AnonymizerError


class Cancelled(AnonymizerError):
    """Cooperative cancellation reached a safe workflow boundary."""


def _bundled_model() -> Path | None:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "model"
    return root if root.is_dir() else None


def _unique_child(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        candidate = parent / f"{prefix}-{uuid.uuid4().hex[:12]}"
        if not candidate.exists():
            return candidate
    raise AnonymizerError("Could not allocate a unique local output directory.")


def eta_range(
    samples: list[tuple[float, int]], completed: int, total: int, *, baseline_rate: float = 0
) -> tuple[float, float] | None:
    """Conservative planning range over pooled timing windows, not a confidence CI.

    Weight durations and completed counts together, so tiny batches do not count
    as much as large ones. Variability widens only the upper end: the lower end
    cannot fall below observed mean throughput or the supplied stage baseline.
    """
    samples = samples[-7:]
    if total <= 0 or completed < 0 or completed >= total or len(samples) < 2:
        return None
    if samples[-1][1] != completed or samples[-1][0] - samples[0][0] < 10:
        return None
    costs = []
    counts = []
    for (start, before), (end, after) in pairwise(samples):
        if end <= start or after <= before or before < 0 or after > total:
            return None
        costs.append((end - start) / (after - before))
        counts.append(after - before)
    weights = [count * 0.8 ** age for count, age in zip(counts, reversed(range(len(costs))))]
    weight = sum(weights)
    mean = sum(cost * w for cost, w in zip(costs, weights)) / weight
    variance = sum(w * (cost - mean) ** 2 for cost, w in zip(costs, weights)) / weight
    margin = max(mean * 0.2, math.sqrt(variance))
    observed_rate = (samples[-1][0] - samples[0][0]) / (samples[-1][1] - samples[0][1])
    lower = max(mean, observed_rate, baseline_rate)
    remaining = total - completed
    return lower * remaining, max(lower * 1.2, mean + margin) * remaining


class WindowedEta:
    """Publish at most once per ten seconds; pool all progress between updates."""

    def __init__(self):
        self._origin: tuple[float, int] | None = None
        self._points: list[tuple[float, int]] = []
        self._last_count = -1
        self._observations = 0
        self._displayed: tuple[float, float] | None = None

    def update(self, now: float, completed: int, total: int) -> tuple[float, float] | None:
        if total <= 0 or completed < 0 or completed >= total:
            self._displayed = None
            return None
        if self._origin is None:
            self._origin = (now, completed)
            self._points = [self._origin]
            self._last_count = completed
            return None
        if completed <= self._last_count:
            return self._displayed
        self._last_count = completed
        self._observations += 1
        if now - self._points[-1][0] < 10 or self._observations < 4:
            return self._displayed
        self._points = [*self._points, (now, completed)][-7:]
        baseline = (now - self._origin[0]) / (completed - self._origin[1])
        self._displayed = eta_range(self._points, completed, total, baseline_rate=baseline)
        return self._displayed


def eta_seconds(samples: list[tuple[float, int]], completed: int, total: int) -> float | None:
    bounds = eta_range(samples, completed, total)
    return None if bounds is None else sum(bounds) / 2


def format_eta(seconds: float | tuple[float, float] | None) -> str:
    if seconds is None:
        return "Estimating..."
    low, high = seconds if isinstance(seconds, tuple) else (seconds, seconds)
    if not math.isfinite(low) or not math.isfinite(high):
        return "Estimating..."
    low, high = sorted((max(0, low), max(0, high)))
    if high < 60:
        return "Less than 1 min remaining"
    lower, upper = math.ceil(low / 60), math.ceil(high / 60)
    if lower == 0:
        return f"Up to about {upper} min remaining"
    if lower == upper:
        return f"About {upper} min remaining"
    return f"About {lower}-{upper} min remaining"


def format_elapsed(seconds: float) -> str:
    hours, rest = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"Elapsed {hours}:{minutes:02d}:{seconds:02d}"


class _DeferredDetector:
    """Construct the production detector in the worker, loading weights on detect."""

    def __init__(self, policy: dict, model_path: Path | None, progress: Callable[..., None]):
        self.policy = policy
        self.model_path = model_path
        self._progress_callback = progress
        self._inner = None

    def _get(self):
        if self._inner is None:
            from .cli import offline_environment
            from .detection import Detector

            offline_environment()
            self._inner = Detector(self.policy, self.model_path, rules_only=False)
            self._inner.progress = self._segment_progress
        return self._inner

    def _segment_progress(self, completed: int, total: int) -> None:
        self._progress_callback("processing", completed, total)

    @property
    def metadata(self):
        return self._get().metadata

    def detect(self, segments):
        self._progress_callback("model_loading", 0, 0)
        detector = self._get()
        detector._ensure_model()
        self._progress_callback("processing", 0, len(segments))
        findings = detector.detect(segments)
        self._progress_callback("creating_review_files", len(segments), len(segments))
        return findings


class DesktopController:
    """Threaded backend used by the native UI and headless smoke tests."""

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        workflow_module: Any = workflow,
        detector_factory: Callable[[dict, Path | None, Callable[..., None]], Any] | None = None,
    ):
        self.model_path = Path(model_path) if model_path is not None else _bundled_model()
        self.workflow = workflow_module
        self.detector_factory = detector_factory or _DeferredDetector
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._run: Path | None = None
        self._export_path: Path | None = None
        self._status: dict[str, Any] | None = None
        self._status_action: str | None = None
        self._progress_stage = None
        self._progress_total = 0
        self._progress_last_completed = -1
        self._eta = WindowedEta()
        self._started_at = 0.0

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def run_path(self) -> Path | None:
        return self._run

    @property
    def status(self) -> dict[str, Any] | None:
        return self._status

    @property
    def export_path(self) -> Path | None:
        return self._export_path

    def messages(self):
        while True:
            try:
                yield self.events.get_nowait()
            except queue.Empty:
                return

    def clear_selection(self) -> None:
        """Detach the prior run without modifying any files on disk."""
        if self.busy:
            raise AnonymizerError("Another operation is still running.")
        self._run = self._status = self._export_path = None
        self._status_action = None
        # A finished worker may still have undelivered result/progress events.
        list(self.messages())

    def _start(
        self,
        action: str,
        target: Callable[[], dict[str, Any] | None],
        *,
        reset_selection: bool = False,
    ) -> None:
        if self.busy:
            raise AnonymizerError("Another operation is still running.")
        if reset_selection:
            self._status = None
            self._run = None
            self._export_path = None
        self._cancel_event.clear()
        self._progress_stage = None
        self._progress_total = 0
        self._progress_last_completed = -1
        self._eta = WindowedEta()
        self._started_at = time.monotonic()
        self.events.put({"kind": "started", "action": action})

        def worker() -> None:
            success = False
            try:
                result = target()
                success = True
                if result is not None:
                    if action != "export":
                        if action == "import" and result.get("import_result") != "already_applied":
                            # Earlier exports remain on disk, but must not be
                            # presented as exports of the newly reviewed state.
                            self._export_path = None
                        self._status = result
                    elif result and result.get("export_directory"):
                        self._export_path = Path(result["export_directory"])
                    self.events.put(
                        {
                            "kind": "result",
                            "action": action,
                            "result": result,
                            "status": self._status,
                        }
                    )
            except Cancelled:
                self.events.put({"kind": "cancelled", "action": action})
            except AnonymizerError as exc:
                self.events.put({"kind": "error", "action": action, "message": str(exc)})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                del exc
                self.events.put(
                    {
                        "kind": "error",
                        "action": action,
                        "message": "The local input or run state could not be processed.",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - sanitize unexpected worker failures
                # Model/package failures must not leak tracebacks or source
                # values into the UI.  The detailed exception remains local to
                # the worker and is deliberately not logged.
                del exc
                self.events.put(
                    {
                        "kind": "error",
                        "action": action,
                        "message": "The local operation failed. Check the model, input, and run folder.",
                    }
                )
            finally:
                if not success and action in {"process", "select_run"}:
                    self._status = None
                    self._run = None
                self.events.put({"kind": "finished", "action": action, "success": success})

        self._thread = threading.Thread(target=worker, name="transcript-anonymizer", daemon=True)
        self._thread.start()

    def _check_cancel(self) -> None:
        if self._cancel_event.is_set():
            raise Cancelled("Cancellation requested; no partial revision was approved.")

    def _progress(self, stage: str, completed: int, total: int) -> None:
        self._check_cancel()
        now = time.monotonic()
        if (
            stage != self._progress_stage
            or total != self._progress_total
            or completed < self._progress_last_completed
        ):
            self._eta = WindowedEta()
        if stage != self._progress_stage or total != self._progress_total:
            self._progress_stage, self._progress_total = stage, total
        self._progress_last_completed = completed
        self.events.put(
            {
                "kind": "progress",
                "stage": stage,
                "completed": completed,
                "total": total,
                "eta": self._eta.update(now, completed, total),
                "elapsed_s": round(now - self._started_at, 1),
            }
        )

    def start_process(
        self, source: Path, output_folder: Path, policy_path: Path | None = None
    ) -> None:
        source, output_folder = Path(source), Path(output_folder)

        def process() -> dict[str, Any]:
            self._check_cancel()
            policy = self._load_policy(policy_path)
            self._check_cancel()
            run = _unique_child(output_folder, "run")
            self._run = run
            from .cli import offline_environment

            offline_environment()
            detector = self.detector_factory(policy, self.model_path, self._progress)
            self._check_cancel()
            return self.workflow.prepare(source, run, policy, detector)

        self._start("process", process, reset_selection=True)

    def select_run(self, run: Path) -> None:
        run = Path(run)

        def select() -> dict[str, Any]:
            self._check_cancel()
            self._run = run
            return self.workflow.status(run)

        self._start("select_run", select, reset_selection=True)

    def import_workbook(self, workbook: Path) -> None:
        workbook = Path(workbook)

        def importing() -> dict[str, Any]:
            self._check_cancel()
            if self._run is None:
                raise AnonymizerError("Select an existing run before importing a workbook.")
            self._progress("importing_decisions", 0, 0)
            state, _ = self.workflow._load(self._run)
            from .cli import offline_environment

            offline_environment()
            detector = self.detector_factory(state["policy"], self.model_path, self._progress)
            self._check_cancel()
            return self.workflow.import_review(self._run, workbook, detector)

        self._start("import", importing)

    def export(self, output_folder: Path) -> None:
        output_folder = Path(output_folder)

        def exporting() -> dict[str, Any]:
            self._check_cancel()
            if self._run is None:
                raise AnonymizerError("Select a prepared run before exporting.")
            if not self._status or not self._status.get("signed_off"):
                raise AnonymizerError("Export is available only after workbook sign-off.")
            destination = _unique_child(output_folder, "export")
            self._check_cancel()
            self._progress("exporting", 0, 0)
            return self.workflow.export(self._run, destination)

        self._start("export", exporting)

    def cancel(self) -> bool:
        if not self.busy:
            return False
        self._cancel_event.set()
        self.events.put({"kind": "cancel_requested"})
        return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @staticmethod
    def _load_policy(path: Path | None) -> dict:
        from .detection import load_policy

        return load_policy(path)


def _open_local(path: Path) -> None:
    if not path.is_file() and not path.is_dir():
        raise AnonymizerError("The requested local review file is unavailable.")
    if sys.platform == "win32":
        try:
            os.startfile(path)  # type: ignore[attr-defined,no-untyped-call]
        except OSError as exc:
            raise AnonymizerError(
                "Could not open the local review file with the system viewer."
            ) from exc
        return
    elif sys.platform == "darwin":
        command = ["open", str(path)]
    else:
        command = ["xdg-open", str(path)]
    try:
        subprocess.Popen(command, start_new_session=True)
    except OSError as exc:
        raise AnonymizerError(
            "Could not open the local review file with the system viewer."
        ) from exc


class DesktopApp:
    """Native single-window Tk frontend."""

    def __init__(self, root=None, controller: DesktopController | None = None):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root or tk.Tk()
        self.controller = controller or DesktopController()
        self.root.title("Offline Transcript PII Review")
        self.root.minsize(760, 430)
        self._output_auto = True
        self.source = tk.StringVar()
        self.policy = tk.StringVar()
        self.output = tk.StringVar()
        self.run_selection = tk.StringVar()
        self.selected_run_text = tk.StringVar(value="No existing run selected")
        self.status_text = tk.StringVar(value="No run selected")
        self.progress_text = tk.StringVar(value="Ready")
        self.elapsed_text = tk.StringVar(value="")
        self.eta_text = tk.StringVar(value="")
        self.next_step_text = tk.StringVar(
            value="Next: choose a source DOCX. Review folders contain sensitive source, context, and workbook files and stay local."
        )
        self._action_buttons = []
        self._controls = []
        self._close_requested = False
        self._build()
        self._selected_source = ""
        self.source.trace_add("write", self._source_changed)
        self._set_busy(False)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._drain)

    def _build(self):
        ttk = self.ttk
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        self._path_row(frame, 0, "Source DOCX", self.source, self._browse_source, "file")
        self._path_row(frame, 1, "Review folder", self.output, self._browse_output, "folder")
        self.advanced_frame = ttk.LabelFrame(frame, text="Advanced", padding=8)
        self._path_row(
            self.advanced_frame, 0, "Policy (optional)", self.policy, self._browse_policy, "file"
        )
        self.advanced_visible = False
        self.advanced_toggle = ttk.Button(
            frame, text="Show advanced", command=self._toggle_advanced
        )
        self.advanced_toggle.grid(row=2, column=0, sticky="w", pady=4)
        self._controls.append(self.advanced_toggle)
        self.resume_button = ttk.Button(
            frame, text="Resume existing review...", command=self._browse_run
        )
        self.resume_button.grid(row=2, column=1, sticky="e", pady=4)
        self._controls.append(self.resume_button)
        ttk.Label(frame, textvariable=self.selected_run_text, wraplength=700).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        ttk.Label(frame, textvariable=self.status_text).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(14, 4)
        )
        ttk.Label(frame, textvariable=self.next_step_text, wraplength=700).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(frame, textvariable=self.progress_text, wraplength=700).grid(
            row=8, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(frame, textvariable=self.elapsed_text).grid(
            row=9, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(frame, textvariable=self.eta_text).grid(row=9, column=2, sticky="e")
        prepare_buttons = ttk.Frame(frame)
        prepare_buttons.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(18, 0))
        self.process_button = ttk.Button(
            prepare_buttons, text="Prepare review", command=self._process
        )
        self.process_button.pack(side="left")
        self.cancel_button = ttk.Button(
            prepare_buttons, text="Cancel", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=6)
        buttons = ttk.Frame(frame)
        buttons.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for label, command in (
            ("Open workbook", self._open_workbook),
            ("Open candidate", self._open_candidate),
            ("Import saved XLSX", self._import),
            ("Open export folder", self._open_export),
        ):
            button = ttk.Button(buttons, text=label, command=command)
            button.pack(side="left", padx=6)
            self._action_buttons.append(button)
            if label == "Open workbook":
                self.open_workbook_button = button
            elif label == "Open candidate":
                self.open_candidate_button = button
            elif label == "Import saved XLSX":
                self.import_button = button
            elif label == "Open export folder":
                self.open_export_button = button
        self.export_button = ttk.Button(
            buttons, text="Export signed-off", command=self._export, state="disabled"
        )
        self.export_button.pack(side="right")
        self._action_buttons.append(self.export_button)

    def _path_row(self, parent, row, label, variable, callback, kind):
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = self.ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        button = self.ttk.Button(parent, text="Browse...", command=callback)
        button.grid(row=row, column=2, pady=4)
        self._controls.extend((entry, button))
        if kind == "folder" and variable is self.output:
            entry.bind("<Key>", lambda _event: setattr(self, "_output_auto", False))

    def _toggle_advanced(self):
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
            self.advanced_toggle.configure(text="Hide advanced")
        else:
            self.advanced_frame.grid_remove()
            self.advanced_toggle.configure(text="Show advanced")

    def _browse_source(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("Word documents", "*.docx")])
        if path:
            self.source.set(path)
            if self._output_auto:
                self.output.set(str(Path(path).parent / "Transcript review"))

    def _source_changed(self, *_args):
        value = self.source.get()
        if value == self._selected_source:
            return
        if self.controller.busy:
            self.source.set(self._selected_source)
            return
        self._selected_source = value
        self.controller.clear_selection()
        self._clear_selection_display()
        self.elapsed_text.set("")
        self.progress_text.set("Ready")
        self.next_step_text.set(
            "Next: choose the review folder, then click Prepare review."
            if value else "Next: choose a source DOCX."
        )
        if value and self._output_auto:
            self.output.set(str(Path(value).parent / "Transcript review"))
        self._set_busy(False)

    def _browse_policy(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            filetypes=[("JSON policy", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.policy.set(path)

    def _browse_output(self):
        from tkinter import filedialog

        path = filedialog.askdirectory()
        if path:
            self._output_auto = False
            self.output.set(path)

    def _browse_run(self):
        from tkinter import filedialog

        path = filedialog.askdirectory()
        if path:
            self.run_selection.set(path)
            self.selected_run_text.set(f"Selected review workspace: {path}")
            if not self.output.get():
                self._output_auto = False
                self.output.set(str(Path(path).parent))
            self.controller.select_run(Path(path))
            self._set_busy(True)

    def _process(self):
        if not self.source.get() or not self.output.get():
            self.progress_text.set("Choose a source DOCX and review folder first.")
            return
        try:
            self.controller.start_process(
                Path(self.source.get()),
                Path(self.output.get()),
                Path(self.policy.get()) if self.policy.get() else None,
            )
            self.next_step_text.set(
                "Next: wait for processing to finish. You can cancel safely; cancellation may wait for the current model chunk."
            )
            self._set_busy(True)
        except AnonymizerError as exc:
            self.progress_text.set(str(exc))

    def _cancel(self):
        if self.controller.cancel():
            self.next_step_text.set(
                "Next: wait until cancellation finishes, then check the run status."
            )
            self.progress_text.set("Cancellation pending; finishing the safe operation...")

    def _import(self):
        from tkinter import filedialog

        if self.controller.run_path is None:
            self.progress_text.set("Select an existing run first.")
            return
        current_workbook = (
            self.controller.status.get("workbook") if self.controller.status else None
        )
        picker_options = {"filetypes": [("Excel workbooks", "*.xlsx")]}
        if current_workbook:
            workbook_path = Path(current_workbook)
            picker_options.update(
                {"initialdir": str(workbook_path.parent), "initialfile": workbook_path.name}
            )
        path = filedialog.askopenfilename(**picker_options)
        if path:
            try:
                self.controller.import_workbook(Path(path))
                self._set_busy(True)
            except AnonymizerError as exc:
                self.progress_text.set(str(exc))

    def _export(self):
        if not self.output.get():
            if self.controller.run_path is None:
                self.progress_text.set("Select a review folder before exporting.")
                return
            self.output.set(str(self.controller.run_path.parent))
        try:
            self.controller.export(Path(self.output.get()))
            self.next_step_text.set(
                "Next: wait for export, then open only the clean export folder for transfer."
            )
            self._set_busy(True)
        except AnonymizerError as exc:
            self.progress_text.set(str(exc))

    def _open_workbook(self):
        self._open_status_file("workbook")

    def _open_candidate(self):
        self._open_status_file("candidate")

    def _open_export(self):
        if self.controller.export_path is None:
            self.progress_text.set("Export the signed-off candidate first.")
            return
        try:
            _open_local(self.controller.export_path)
        except AnonymizerError as exc:
            self.progress_text.set(str(exc))

    def _open_status_file(self, key):
        status = self.controller.status
        if not status:
            self.progress_text.set("Select or process a run first.")
            return
        try:
            _open_local(Path(status[key]))
        except AnonymizerError as exc:
            self.progress_text.set(str(exc))

    def _set_busy(self, busy):
        self.process_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        for button in self._action_buttons:
            if button is self.export_button:
                enabled = bool(self.controller.status and self.controller.status.get("signed_off"))
                button.configure(state="disabled" if busy or not enabled else "normal")
            elif button is self.open_export_button:
                enabled = self.controller.export_path is not None
                button.configure(state="disabled" if busy or not enabled else "normal")
            elif button in {
                self.open_workbook_button,
                self.open_candidate_button,
                self.import_button,
            }:
                enabled = (
                    self.controller.run_path is not None and self.controller.status is not None
                )
                button.configure(state="disabled" if busy or not enabled else "normal")
            else:
                button.configure(state="disabled" if busy else "normal")
        for control in self._controls:
            control.configure(state="disabled" if busy else "normal")

    def _drain(self):
        for message in self.controller.messages():
            kind = message["kind"]
            if kind == "started":
                self._ui_started_at = time.monotonic()
                self.elapsed_text.set(format_elapsed(0))
                self.eta_text.set(
                    "Estimating..." if message.get("action") in {"process", "import"} else ""
                )
            elif kind == "progress":
                stage = message["stage"]
                if stage == "model_loading":
                    self.status_text.set("Status: Loading model")
                    self.progress["value"] = 2
                    self.progress_text.set("Loading local model...")
                    self.eta_text.set("Estimating...")
                    self.next_step_text.set(
                        "Next: wait while the local model loads. No network access is used."
                    )
                elif stage == "creating_review_files":
                    self.status_text.set("Status: Creating review files")
                    self.progress["value"] = 95
                    self.progress_text.set("Creating review files...")
                    self.eta_text.set("Almost done")
                    self.next_step_text.set(
                        "Next: open the workbook in the sensitive review workspace."
                    )
                elif stage == "importing_decisions":
                    self.progress_text.set("Importing saved workbook decisions...")
                    self.eta_text.set("")
                    self.next_step_text.set(
                        "Next: wait for corrections to be checked and the review files regenerated."
                    )
                elif stage == "exporting":
                    self.progress_text.set("Writing clean export...")
                    self.eta_text.set("")
                    self.next_step_text.set("Next: open only the clean export folder for transfer.")
                else:
                    self.status_text.set("Status: Processing")
                    self.next_step_text.set(
                        "Next: wait for transcript processing to finish. You can cancel safely; "
                        "cancellation may wait for the current model chunk."
                    )
                    completed, total = message["completed"], message["total"]
                    self.progress["value"] = 5 + (85 * completed / total if total else 0)
                    self.progress_text.set(
                        f"Processing {completed} of {total} segments"
                    )
                    self.eta_text.set("Processing: " + format_eta(message["eta"]))
            elif kind == "cancel_requested":
                self.progress_text.set("Cancellation pending; finishing the safe operation...")
                self.eta_text.set("")
            elif kind == "result":
                result = message["result"]
                if message.get("action") == "export":
                    self.progress_text.set(
                        f"Export ready: {result.get('export_directory', 'local export folder')}"
                    )
                elif result:
                    self._apply_status(result, message.get("action"))
            elif kind == "cancelled":
                if message.get("action") in {"process", "select_run"}:
                    self._clear_selection_display()
                self.progress_text.set("Cancelled. No partial revision was approved.")
                self.next_step_text.set(
                    "Next: select the source DOCX and start a new process, or select an existing run."
                )
            elif kind == "error":
                if message.get("action") in {"process", "select_run"}:
                    self._clear_selection_display()
                self.progress_text.set(message["message"])
                self.next_step_text.set(
                    "Next: follow the message above, correct the local input or run selection, and try again."
                )
            elif kind == "finished":
                if hasattr(self, "_ui_started_at"):
                    self.elapsed_text.set(format_elapsed(time.monotonic() - self._ui_started_at))
                self.eta_text.set("")
                if message.get("success"):
                    self.progress["value"] = 100
                self._set_busy(False)
                if self._close_requested:
                    self.root.destroy()
                    return
        if self.controller.busy and hasattr(self, "_ui_started_at"):
            self.elapsed_text.set(format_elapsed(time.monotonic() - self._ui_started_at))
        self.root.after(100, self._drain)

    def _clear_selection_display(self):
        """Remove stale run affordances after a failed new selection or prepare."""
        self._status = None
        self.run_selection.set("")
        self.selected_run_text.set("No existing run selected")
        self.status_text.set("No run selected")
        self.progress["value"] = 0
        self.eta_text.set("")

    def _apply_status(self, result, action=None):
        self._status = result
        self._status_action = action
        if result.get("signed_off"):
            self.status_text.set("Status: Signed off")
        else:
            self.status_text.set("Status: Pending review")
        if result.get("run_id"):
            candidate = Path(result.get("candidate", ""))
            if len(candidate.parents) >= 3:
                self.run_selection.set(str(candidate.parents[2]))
                self.selected_run_text.set(f"Selected review workspace: {candidate.parents[2]}")
        self.progress_text.set(
            "Ready for review"
            if not result.get("signed_off")
            else "Signed-off candidate ready to export"
        )
        if result.get("signed_off"):
            self.next_step_text.set(
                "Next: export the signed-off candidate, then open only the clean export folder for transfer."
            )
        elif (
            action == "import"
            and result.get("import_result") == "candidate_changed_signoff_cleared"
        ):
            self.next_step_text.set(
                "Sign-off was cleared because the candidate changed. Open the updated workbook and review it. When complete, enter your name in Signed off by, save and close Excel, then import again."
            )
        else:
            self.next_step_text.set(
                "Next: open the review workbook and check the proposed replacements and ranked passages. When review is complete, enter your name in Signed off by. Save and close Excel, then import the saved workbook."
            )
        self.eta_text.set("")
        self._set_busy(self.controller.busy)

    def _close(self):
        if self.controller.busy:
            self._close_requested = True
            self.controller.cancel()
            self.progress_text.set(
                "Cancellation pending; the window will close when the worker exits safely..."
            )
            return
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline transcript PII review desktop app")
    parser.add_argument("--model", type=Path, help="Local model directory; no downloads")
    parser.add_argument("--self-test", type=Path, help="Run the synthetic workflow into a new folder and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        from .selftest import run
        run(args.self_test, args.model or _bundled_model())
        return 0
    import tkinter as tk

    root = tk.Tk()
    DesktopApp(root=root, controller=DesktopController(model_path=args.model)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
