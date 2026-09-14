"""Transactional local review workflow. Only allowlisted signed bytes can leave a run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .detection import rank_passages
from .documents import read_docx, validate_output, write_docx
from .errors import AnonymizerError
from .ranking import canonical_occurrence_id, rank_retained_passages, resolve_findings
from .workbook import read_workbook, write_workbook

_CURRENT_OVERLAP_RESOLVER_VERSION = 2
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _review_filename(source: Path) -> str:
    """Keep the source recognizable while bounding Windows filename hazards."""
    original = source.stem
    stem = _UNSAFE_FILENAME.sub("_", original).strip(" .") or "Transcript"
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", stem.split(".")[0]):
        stem = "_" + stem
    while len(stem.encode("utf-8")) > 120:
        stem = stem[:-1]
    if stem != original:
        stem = stem.rstrip(" .") + "-" + hashlib.sha256(source.name.encode()).hexdigest()[:8]
    return f"{stem} - review.xlsx"


def _workbook_filename(state: dict) -> str:
    # Existing revisions keep their original name; no approval migration.
    name = state.get("workbook_filename", "review.xlsx")
    if (
        not isinstance(name, str)
        or _UNSAFE_FILENAME.search(name)
        or not name.endswith(".xlsx")
        or len(name.encode("utf-8")) > 200
        or name.startswith((".", " "))
        or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", name.split(".")[0])
    ):
        raise AnonymizerError("Run workbook filename is invalid.")
    return name


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def run_lock(run: Path):
    lock = run / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise AnonymizerError(
            "Run is locked. Check for an active writer before recovery."
        ) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _load(run: Path) -> tuple[dict, Path]:
    try:
        pointer = json.loads((run / "current.json").read_bytes())
        name = pointer["directory"]
        if not re.fullmatch(r"r\d{6}-[0-9a-f]{12}", name):
            raise ValueError
        folder = run / "revisions" / name
        raw = (folder / "state.json").read_bytes()
        if digest(raw) != pointer["sha256"]:
            raise ValueError
        state = json.loads(raw)
        if state["schema_version"] != 1:
            raise ValueError
        _workbook_filename(state)
        if file_hash(folder / "candidate.docx") != state["candidate_sha256"]:
            raise AnonymizerError(
                "Candidate bytes changed. Restore the reviewed revision or reprepare."
            )
        return state, folder
    except (OSError, KeyError, ValueError, TypeError):
        raise AnonymizerError("Run state is missing, incompatible or inconsistent.") from None


def _commit(run: Path, state: dict, rendered: list[dict], previous: Path | None = None):
    revisions = run / "revisions"
    revisions.mkdir(exist_ok=True, mode=0o700)
    folder = Path(tempfile.mkdtemp(prefix="pending-", dir=revisions))
    try:
        candidate = folder / "candidate.docx"
        if previous is None:
            write_docx(rendered, candidate)
        else:
            shutil.copyfile(previous / "candidate.docx", candidate)
        validate_output(candidate)
        state["candidate_sha256"] = file_hash(candidate)
        if state.get("signed_off_by"):
            state["signoff"] = {
                "name": state["signed_off_by"],
                "time": now(),
                "candidate_sha256": state["candidate_sha256"],
                "revision": state["revision"],
                "policy_sha256": state["policy_sha256"],
            }
        else:
            state["signoff"] = None
        state["view"] = _view(state, rendered)
        write_workbook(state["view"], folder / _workbook_filename(state))
        raw = encoded(state)
        with (folder / "state.json").open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        final = revisions / f"r{state['revision']:06d}-{uuid.uuid4().hex[:12]}"
        folder.rename(final)
        pointer = {"directory": final.name, "sha256": digest(raw)}
        pending_pointer = run / ".current.pending"
        with pending_pointer.open("wb") as handle:
            handle.write(encoded(pointer))
            handle.flush()
            os.fsync(handle.fileno())
        pending_pointer.replace(run / "current.json")
        return status(run)
    finally:
        # Only our unpublished staging folder is removed on failure.
        if folder.exists():
            shutil.rmtree(folder)


def _finding_id(finding: dict) -> str:
    return canonical_occurrence_id(finding)


def _require_current_overlap_resolver(state: dict) -> None:
    if state.get("overlap_resolver_version") != _CURRENT_OVERLAP_RESOLVER_VERSION:
        raise AnonymizerError(
            "This run predates the coverage-preserving overlap resolver; "
            "prepare a new run and review it again before importing or exporting."
        )


def _reconcile(state: dict):
    texts = {s["id"]: s["text"] for s in state["segments"]}
    all_findings = []
    for f in state["findings"]:
        f = deepcopy(f)
        try:
            text = texts[f["segment_id"]]
            if (
                not isinstance(f["start"], int)
                or not isinstance(f["end"], int)
                or not 0 <= f["start"] < f["end"] <= len(text)
                or text[f["start"] : f["end"]] != f["text"]
            ):
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise AnonymizerError("Detector returned an invalid source span.") from None
        f["id"] = _finding_id(f)
        all_findings.append(f)
    # Resolve replacement candidates deterministically; retain review-only and
    # overlap-suppressed candidates locally for the ranked review queue.
    state["suppressed"] = []
    replacement_findings = [f for f in all_findings if not f.get("review_only", False)]
    state["suppressed"].extend(
        {**f, "reason": "review_only"} for f in all_findings if f.get("review_only", False)
    )
    resolver_version = state.get("overlap_resolver_version", 1)
    if (
        isinstance(resolver_version, bool)
        or not isinstance(resolver_version, int)
        or resolver_version not in {1, 2}
    ):
        raise AnonymizerError("Run uses an unsupported finding resolver version.")
    selected, overlap_suppressed = resolve_findings(
        replacement_findings, preserve_coverage=resolver_version >= 2
    )
    state["suppressed"].extend(overlap_suppressed)
    selected.sort(key=lambda f: (f["segment_id"], f["start"], f["id"]))
    old = state.get("groups", {})
    groups = {}
    for f in selected:
        key = [f["category"], f.get("entity_key") or f["text"].casefold()]
        gid = "g" + digest(encoded(key))[:20]
        f["group_id"] = gid
        if gid not in groups:
            if gid in old:
                groups[gid] = {**old[gid], "occurrence_ids": [], "aliases": []}
            else:
                category = re.sub(r"[^A-Z0-9_]", "", f["category"].upper()) or "PII"
                count = state["token_counters"].get(category, 0) + 1
                state["token_counters"][category] = count
                token = f"{category}_{count:03d}"
                while any(token in text for text in texts.values()):
                    count += 1
                    state["token_counters"][category] = count
                    token = f"{category}_{count:03d}"
                groups[gid] = {
                    "id": gid,
                    "category": f["category"],
                    "token": token,
                    "action": "accept",
                    "replacement": "",
                    "note": "",
                    "occurrence_ids": [],
                    "aliases": [],
                }
        group = groups[gid]
        group["occurrence_ids"].append(f["id"])
        if f["text"] not in group["aliases"]:
            group["aliases"].append(f["text"])
    state["selected"] = selected
    state["groups"] = groups
    valid = {f["id"] for f in selected}
    state["occurrence_decisions"] = {
        k: v for k, v in state.get("occurrence_decisions", {}).items() if k in valid
    }


def _render(state: dict) -> list[dict]:
    by_segment = {}
    for f in state["selected"]:
        by_segment.setdefault(f["segment_id"], []).append(f)
    result = []
    for segment in state["segments"]:
        text = segment["text"]
        for f in sorted(by_segment.get(segment["id"], []), key=lambda x: -x["start"]):
            group = state["groups"][f["group_id"]]
            decision = state["occurrence_decisions"].get(f["id"], {"action": "inherit"})
            if decision["action"] == "inherit":
                decision = group
            action = decision["action"]
            replacement = {
                "retain": f["text"],
                "remove": "",
                "accept": group["token"],
                "replace": decision.get("replacement", ""),
            }[action]
            text = text[: f["start"]] + replacement + text[f["end"] :]
        result.append({**segment, "text": text})
    return result


def _masked_intervals(state: dict) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    for finding in state["selected"]:
        group = state["groups"][finding["group_id"]]
        decision = state["occurrence_decisions"].get(finding["id"], {"action": "inherit"})
        if decision["action"] == "inherit":
            decision = group
        action = decision["action"]
        replacement = {
            "retain": finding["text"],
            "remove": "",
            "accept": group["token"],
            "replace": decision.get("replacement", ""),
        }[action]
        # A replacement is only considered masked when it no longer carries
        # the original finding text.  This keeps ranking evidence visible for
        # a reviewer-entered replacement that merely appends to the source.
        replacement_masks = action in {"accept", "remove"} or (
            action == "replace" and finding["text"].casefold() not in replacement.casefold()
        )
        if replacement_masks:
            intervals.append(
                {
                    "segment_id": finding["segment_id"],
                    "start": finding["start"],
                    "end": finding["end"],
                }
            )
    return intervals


def _view(state: dict, rendered: list[dict]) -> dict:
    originals = {s["id"]: s["text"] for s in state["source_segments"]}
    working = {s["id"]: s["text"] for s in state["segments"]}
    findings = {f["id"]: f for f in state["selected"]}
    groups = []
    for group in state["groups"].values():
        first = findings[group["occurrence_ids"][0]]
        groups.append(
            {k: group[k] for k in ("id", "category", "token", "action", "replacement", "note")}
            | {
                "source": group["aliases"][0],
                "aliases": " | ".join(group["aliases"]),
                "count": len(group["occurrence_ids"]),
                "context": working[first["segment_id"]],
            }
        )
    occurrences = []
    for f in state["selected"]:
        occurrences.append(
            {
                "id": f["id"],
                "group_id": f["group_id"],
                "segment_id": f["segment_id"],
                "source": f["text"],
                "context": working[f["segment_id"]],
                "action": "inherit",
                "replacement": "",
                "note": "",
            }
            | state["occurrence_decisions"].get(f["id"], {})
        )
    retained = {s["id"]: s["text"] for s in rendered}
    passages = []
    if state.get("review_ranker_version", 1) >= 2:
        rankings = rank_retained_passages(
            state["segments"],
            rendered,
            state["findings"],
            _masked_intervals(state),
            sample_size=state.get("review_sample_size", 5),
        )
    else:
        rankings = rank_passages(rendered, [])
    for ranked in rankings:
        sid = ranked["id"]
        evidence = state.get("passage_reviews", {}).get(sid, {})
        outcome = evidence.get("outcome", "uninspected")
        if evidence.get("text_sha256") != digest(retained[sid].encode()):
            outcome = "uninspected"
        reasons = ranked.get("reasons", [])
        passages.append(
            {
                "id": sid,
                "source": originals[sid],
                "text": retained[sid],
                "score": ranked["score"],
                "reasons": "; ".join(reasons) if isinstance(reasons, list) else reasons,
                "outcome": outcome,
                "action": "keep",
                "replacement": "",
                "note": evidence.get("note", ""),
            }
        )
    view = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "revision": state["revision"],
        "candidate_sha256": state["candidate_sha256"],
        "signed_off_by": state.get("signed_off_by", ""),
        "groups": groups,
        "passages": passages,
        "occurrences": occurrences,
    }
    if "source_filename" in state:
        view["source_filename"] = state["source_filename"]
    return view


def prepare(
    source: Path, run: Path, policy: dict, detector, *, review_sample_size: int = 5
) -> dict:
    if (
        not isinstance(review_sample_size, int)
        or isinstance(review_sample_size, bool)
        or not 0 <= review_sample_size <= 100
    ):
        raise AnonymizerError("Review sample size must be between 0 and 100.")
    if run.exists():
        raise AnonymizerError("Run destination already exists; choose a new directory.")
    if hasattr(detector, "policy") and encoded(detector.policy) != encoded(policy):
        raise AnonymizerError("Preparation policy differs from the detector's effective policy.")
    before = file_hash(source)
    document = read_docx(source)
    findings = detector.detect(document["segments"])
    if file_hash(source) != before:
        raise AnonymizerError("Source changed during preparation; retry from a stable input.")
    state = {
        "schema_version": 1,
        "app_version": __version__,
        "run_id": uuid.uuid4().hex,
        "revision": 1,
        "created": now(),
        "source_sha256": before,
        "source_filename": source.name,
        "workbook_filename": _review_filename(source),
        "source_segments": deepcopy(document["segments"]),
        "segments": document["segments"],
        "inventory": document["inventory"],
        "policy": policy,
        "policy_sha256": digest(encoded(policy)),
        "detector": detector.metadata,
        "findings": findings,
        "token_counters": {},
        "occurrence_decisions": {},
        "passage_reviews": {},
        "signed_off_by": "",
        "history": [{"event": "prepare", "time": now()}],
        "imports": [],
        "review_ranker_version": 2,
        "overlap_resolver_version": _CURRENT_OVERLAP_RESOLVER_VERSION,
        "review_sample_size": review_sample_size,
    }
    _reconcile(state)
    run.mkdir(parents=True, mode=0o700)
    with run_lock(run):
        return _commit(run, state, _render(state))


def import_review(run: Path, workbook: Path, detector=None) -> dict:
    with run_lock(run):
        state, previous = _load(run)
        _require_current_overlap_resolver(state)
        workbook_hash = file_hash(workbook)
        if workbook_hash in state["imports"]:
            return {**status(run), "import_result": "already_applied"}
        decisions = read_workbook(workbook, state["view"])
        new = deepcopy(state)
        old_rendered = _render(state)
        old_text = {s["id"]: s["text"] for s in old_rendered}
        changed_targets = set()
        for row in decisions["groups"]:
            group = new["groups"][row["id"]]
            edit = {k: row.get(k, "") for k in ("action", "replacement", "note")}
            if edit["action"] == "replace" and not edit["replacement"].strip():
                raise AnonymizerError("Replacement text is required; use remove for deletion.")
            if any(edit[k] != group[k] for k in ("action", "replacement")):
                changed_targets.update(
                    f["segment_id"] for f in new["selected"] if f["group_id"] == row["id"]
                )
            group.update(edit)
        for row in decisions["occurrences"]:
            edit = {k: row.get(k, "") for k in ("action", "replacement", "note")}
            prior = new["occurrence_decisions"].get(
                row["id"], {"action": "inherit", "replacement": "", "note": ""}
            )
            if edit["action"] == "replace" and not edit["replacement"].strip():
                raise AnonymizerError("Occurrence replacement is empty; use remove for deletion.")
            if any(edit[k] != prior[k] for k in ("action", "replacement")):
                changed_targets.add(
                    next(f["segment_id"] for f in new["selected"] if f["id"] == row["id"])
                )
            new["occurrence_decisions"][row["id"]] = edit
        correction_ids = set()
        for row in decisions["passages"]:
            sid = row["id"]
            if row["action"] != "keep":
                if sid in changed_targets:
                    raise AnonymizerError(
                        f"Conflicting passage and entity edits for {sid}; import separately."
                    )
                if row["action"] == "replace" and not row["replacement"].strip():
                    raise AnonymizerError(f"Passage {sid}: replacement is empty; use remove.")
                correction_ids.add(sid)
                replacement = row["replacement"] if row["action"] == "replace" else ""
                next(s for s in new["segments"] if s["id"] == sid)["text"] = replacement
            elif row["outcome"] == "miss" and sid not in changed_targets:
                raise AnonymizerError(f"Passage {sid}: record a correction for the reported miss.")
            new["passage_reviews"][sid] = {
                "outcome": row["outcome"],
                "note": row.get("note", ""),
                "text_sha256": digest(old_text[sid].encode()),
            }
        if correction_ids or changed_targets:
            if detector is None:
                raise AnonymizerError("Corrections require the original detector for rescanning.")
            if detector.metadata != state["detector"]:
                raise AnonymizerError(
                    "Detector configuration differs from preparation; use the recorded model/policy."
                )
            if (
                hasattr(detector, "policy")
                and digest(encoded(detector.policy)) != state["policy_sha256"]
            ):
                raise AnonymizerError(
                    "Detector policy differs from preparation; use the recorded policy."
                )
            if correction_ids:
                # Detector context can depend on neighboring turns and speaker
                # labels; a correction therefore invalidates the full working
                # document, not only the edited segment.
                new["findings"] = detector.detect(new["segments"])
                _reconcile(new)
            # Scan every newly supplied replacement, not the entire transcript a second time.
            supplied = []
            for i, row in enumerate(decisions["groups"] + decisions["occurrences"]):
                if row["action"] == "replace" and row.get("replacement"):
                    supplied.append(
                        {"id": f"replacement{i}", "text": row["replacement"], "locator": "review"}
                    )
            if supplied and detector.detect(supplied):
                raise AnonymizerError(
                    "A replacement contains detected PII. Use a neutral token or revise the text."
                )
        rendered = _render(new)
        content_changed = [s["text"] for s in rendered] != [s["text"] for s in old_rendered]
        new["signed_off_by"] = "" if content_changed else decisions["signed_off_by"].strip()
        new["revision"] += 1
        new["imports"].append(workbook_hash)
        new["history"].append(
            {
                "event": "review_import",
                "time": now(),
                "base_revision": state["revision"],
                "workbook_sha256": workbook_hash,
                "content_changed": content_changed,
                "decisions": decisions,
            }
        )
        result = _commit(run, new, rendered, None if content_changed else previous)
        result["import_result"] = (
            "candidate_changed_signoff_cleared" if content_changed else "applied"
        )
        return result


def status(run: Path) -> dict:
    state, folder = _load(run)
    view = state["view"]
    return {
        "run_id": state["run_id"],
        "revision": state["revision"],
        "signed_off": bool(state["signoff"]),
        "segments": len(state["segments"]),
        "groups": len(view["groups"]),
        "occurrences": len(view["occurrences"]),
        "inspected_passages": sum(p["outcome"] != "uninspected" for p in view["passages"]),
        "uninspected_passages": sum(p["outcome"] == "uninspected" for p in view["passages"]),
        "sample_recommended": sum("remainder_sample" in p["reasons"] for p in view["passages"]),
        "sample_reviewed": sum(
            "remainder_sample" in p["reasons"] and p["outcome"] != "uninspected"
            for p in view["passages"]
        ),
        "candidate": str(folder / "candidate.docx"),
        "workbook": str(folder / _workbook_filename(state)),
        "detector": state["detector"],
        "overlap_resolver_version": state.get("overlap_resolver_version", 1),
    }


def export(run: Path, destination: Path) -> dict:
    if destination.exists() or destination.resolve().is_relative_to(run.resolve()):
        raise AnonymizerError(
            "Export destination must be new and outside the sensitive run directory."
        )
    with run_lock(run):
        state, folder = _load(run)
        _require_current_overlap_resolver(state)
        signoff = state.get("signoff")
        if (
            not signoff
            or signoff["candidate_sha256"] != state["candidate_sha256"]
            or signoff["revision"] != state["revision"]
            or signoff["policy_sha256"] != state["policy_sha256"]
        ):
            raise AnonymizerError("Current workbook sign-off is required before export.")
        validate_output(folder / "candidate.docx")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
        try:
            target = staging / "transcript.docx"
            shutil.copyfile(folder / "candidate.docx", target)
            if file_hash(target) != signoff["candidate_sha256"]:
                raise AnonymizerError("Candidate changed during export.")
            manifest = {
                "schema_version": 1,
                "application_version": __version__,
                "document_id": state["run_id"],
                "artifacts": {"transcript.docx": {"sha256": file_hash(target)}},
            }
            (staging / "manifest.json").write_bytes(encoded(manifest))
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {"export_directory": str(destination), "files": ["transcript.docx", "manifest.json"]}
