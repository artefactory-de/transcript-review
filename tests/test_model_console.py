"""Model startup must not depend on a UTF-8 console or any console at all."""

import io
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from transcript_anonymizer import detection
from transcript_anonymizer.errors import AnonymizerError


@pytest.mark.parametrize("windowed", [False, True])
def test_model_load_tolerates_legacy_or_absent_console(monkeypatch, tmp_path, windowed):
    expected = object()

    def load(*args, **kwargs):
        # Match the vendor's unconditional Unicode banner, plus the direct
        # stream writes that progress libraries use in a windowed executable.
        print("\U0001f9e0 Model Configuration")
        sys.stdout.flush()
        sys.stderr.write("\U0001f9e0 Loading model\n")
        sys.stderr.flush()
        return expected

    monkeypatch.setattr(detection, "_verify_model_assets", lambda path: None)
    monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(
        AutoExtractor=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None))
    stdout = None if windowed else io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    stderr = None if windowed else io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    with monkeypatch.context() as streams:
        streams.setattr(sys, "stdout", stdout)
        streams.setattr(sys, "stderr", stderr)
        assert detection.Detector._load_model(tmp_path) is expected
        assert sys.stdout is stdout and sys.stderr is stderr


def test_model_load_restores_streams_when_backend_fails(monkeypatch, tmp_path):
    def load(*args, **kwargs):
        raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(detection, "_verify_model_assets", lambda path: None)
    monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(
        AutoExtractor=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None))
    stdout, stderr = sys.stdout, sys.stderr
    with pytest.raises(AnonymizerError, match="could not be loaded") as failure:
        detection.Detector._load_model(tmp_path)
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert sys.stdout is stdout and sys.stderr is stderr


def test_concurrent_model_startup_restores_process_streams(monkeypatch, tmp_path):
    """Overlapping startup redirects must not leave a silent stream installed."""

    entered_a = threading.Event()
    entered_b = threading.Event()
    returned_a = threading.Event()
    calls: list[str] = []

    def load(*args, **kwargs):
        del args, kwargs
        if threading.current_thread().name == "model-a":
            entered_a.set()
            # With no startup lock, model-b enters this context before A exits.
            entered_b.wait(0.2)
            returned_a.set()
        else:
            entered_b.set()
            returned_a.wait(0.5)
            # Ensure A has had a chance to exit before B's context exits.
            time.sleep(0.05)
        calls.append(threading.current_thread().name)
        return object()

    monkeypatch.setattr(detection, "_verify_model_assets", lambda path: None)
    monkeypatch.setitem(
        sys.modules,
        "gliner2",
        SimpleNamespace(AutoExtractor=SimpleNamespace(from_pretrained=load)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda n: None,
            set_num_interop_threads=lambda n: None,
        ),
    )
    stdout, stderr = sys.stdout, sys.stderr
    errors: list[BaseException] = []

    def start():
        try:
            detection.Detector._load_model(tmp_path)
        except Exception as exc:  # noqa: BLE001 - collect worker failures for assertions
            errors.append(exc)

    first = threading.Thread(target=start, name="model-a")
    second = threading.Thread(target=start, name="model-b")
    first.start()
    assert entered_a.wait(1)
    second.start()
    first.join(2)
    second.join(2)
    streams_restored = False
    try:
        assert not first.is_alive() and not second.is_alive()
        assert not errors
        assert sorted(calls) == ["model-a", "model-b"]
        streams_restored = sys.stdout is stdout and sys.stderr is stderr
    finally:
        # A failing assertion must not poison pytest's own capture streams.
        sys.stdout, sys.stderr = stdout, stderr
    assert streams_restored
