import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("stanza_runner", Path(__file__).parents[1] / "scripts" / "benchmarks" / "stanza_runner.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_block_network = _MODULE._block_network
_stanza_findings = _MODULE._stanza_findings
_trusted_numpy_checkpoint_globals = _MODULE._trusted_numpy_checkpoint_globals


class _Entity:
    def __init__(self, entity_type: str, start: int, end: int) -> None:
        self.type = entity_type
        self.start_char = start
        self.end_char = end
        self.text = "Anna" if (start, end) == (0, 4) else ""


class _Document:
    def __init__(self) -> None:
        self.entities = [_Entity("PER", 0, 4), _Entity("ORG", 5, 9), _Entity("LOC", 10, 14)]


class _Pipeline:
    def __call__(self, text: str) -> _Document:
        assert text == "Anna Bank Berlin"
        return _Document()


def test_stanza_adapter_emits_only_person_with_exact_offsets() -> None:
    rows = _stanza_findings(_Pipeline(), [{"id": "d1", "segments": [{"id": "s1", "text": "Anna Bank Berlin"}]}], "conll03")
    assert len(rows) == 1
    assert rows[0][0]["text"] == "Anna"
    assert rows[0][0]["category"] == "person"
    assert rows[0][0]["score"] == 1.0
    assert len(rows[0]) == 1


def test_network_blocker_fails_closed() -> None:
    import socket

    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    _block_network()
    try:
        socket.create_connection(("example.invalid", 443), timeout=0.01)
    except RuntimeError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("network blocker did not reject a connection")
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = originals


def test_stanza_adapter_rejects_malformed_offsets_and_text() -> None:
    class BadOffsetEntity(_Entity):
        def __init__(self) -> None:
            super().__init__("PER", 0, 4)
            self.start_char = 0.0

    class BadTextEntity(_Entity):
        def __init__(self) -> None:
            super().__init__("PER", 0, 4)
            self.text = "Anya"

    class Pipeline:
        def __init__(self, entity: _Entity) -> None:
            self.entity = entity

        def __call__(self, _: str) -> object:
            return type("Document", (), {"entities": [self.entity]})()

    document = [{"id": "d1", "segments": [{"id": "s1", "text": "Anna Bank Berlin"}]}]
    import pytest

    with pytest.raises(ValueError, match="non-integer"):
        _stanza_findings(Pipeline(BadOffsetEntity()), document, "conll03")
    with pytest.raises(ValueError, match="does not match"):
        _stanza_findings(Pipeline(BadTextEntity()), document, "conll03")


def test_numpy_checkpoint_context_restores_safe_globals() -> None:
    import numpy as np
    import torch
    from numpy.core.multiarray import _reconstruct

    original = set(torch.serialization.get_safe_globals())
    with _trusted_numpy_checkpoint_globals(torch):
        safe = set(torch.serialization.get_safe_globals())
        assert (_reconstruct, "numpy.core.multiarray._reconstruct") in safe
        assert np.ndarray in safe
        assert np.dtype in safe
        assert type(np.dtype("float32")) in safe
    assert set(torch.serialization.get_safe_globals()) == original
