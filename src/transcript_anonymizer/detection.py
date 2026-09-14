"""Bounded, offline PII candidate detection.

The detector deliberately returns *candidates*, not an anonymisation decision.  Rules
are deterministic and the optional GLiNER2 pass is kept behind a small adapter so a
missing or broken model can never silently turn into a rules-only run.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.resources
import io
import json
import re
import shutil
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from .errors import AnonymizerError


class _SilentModelConsole(io.TextIOBase):
    """Discard vendor banners without a console, encoding errors or buffering.

    Keep this sink alive: vendor logging handlers may retain their startup stream.
    Application progress and sanitized errors use the restored original streams.
    """

    @property
    def encoding(self):
        return "utf-8"

    def write(self, text):
        return len(text)

    def flush(self):
        pass


_MODEL_CONSOLE = _SilentModelConsole()
_MODEL_CONSOLE_LOCK = threading.RLock()

MODEL_NAME = "fastino/gliner2-privacy-filter-PII-multi"
MODEL_REVISION = "c153999da5f4c509df4322b0c6a1baf3d2c284d7"
_POLICY_KEYS = {"schema_version", "model", "rules", "aliases"}
_MODEL_POLICY_KEYS = {
    "labels",
    "threshold",
    "proposal_threshold",
    "review_threshold",
    "thresholds",
    "label_thresholds",
    "chunk_tokens",
    "chunk_overlap_tokens",
    "name",
    "revision",
}
_RULE_POLICY_KEYS = {"email", "phone", "iban", "customer_id", "depot_id", "account_id"}

# These are intentionally direct PII labels.  Generic business entities (company,
# location, date, amount, product, etc.) must not become redaction candidates merely
# because a model knows how to label them.
DEFAULT_MODEL_LABELS = [
    "person",
    "full_name",
    "first_name",
    "middle_name",
    "last_name",
    "email",
    "phone_number",
    "address",
    "street_address",
    "postal_code",
    "iban",
    "bank_account",
    "account_number",
    "account_id",
    "sensitive_account_id",
    "government_id",
    "national_id_number",
    "passport_number",
    "drivers_license_number",
    "tax_id",
    "tax_number",
    "date_of_birth",
    "username",
    "ip_address",
]

_MODEL_CATEGORY = {
    "person": "person",
    "full_name": "person",
    "first_name": "person",
    "middle_name": "person",
    "last_name": "person",
    "email": "email",
    "phone_number": "phone",
    "address": "address",
    "street_address": "address",
    "postal_code": "address",
    "iban": "iban",
    "bank_account": "account_id",
    "account_number": "account_id",
    "account_id": "account_id",
    "sensitive_account_id": "account_id",
    "government_id": "government_id",
    "national_id_number": "government_id",
    "passport_number": "government_id",
    "drivers_license_number": "government_id",
    "tax_id": "government_id",
    "tax_number": "government_id",
    # A birth date is direct PII; generic dates deliberately remain out of scope.
    "date_of_birth": "date_of_birth",
    "username": "username",
    "ip_address": "ip_address",
}

_DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "model": {
        "labels": list(DEFAULT_MODEL_LABELS),
        "threshold": 0.35,
        # ``threshold`` remains the compatibility spelling.  New policies use
        # separate proposal and review thresholds; review deliberately starts
        # lower so uncertain evidence reaches the ranker.
        "proposal_threshold": 0.35,
        "review_threshold": 0.20,
        "thresholds": {},
        "chunk_tokens": 384,
        "chunk_overlap_tokens": 64,
        "name": MODEL_NAME,
        "revision": MODEL_REVISION,
    },
    "rules": {
        "email": True,
        "phone": True,
        "iban": True,
        "customer_id": True,
        "depot_id": True,
        "account_id": True,
    },
    "aliases": [],
}

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![A-Za-z0-9-])"
)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,5}\)?[ .-/]?)"
    r"(?:\d[ .-/]?){5,13}\d(?!\w)"
)
_IBAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){10,32}(?![A-Z0-9])", re.IGNORECASE)

# The value is a named group so the context word (which is useful to avoid false
# positives) is not itself proposed for replacement.
_CUSTOMER_RE = re.compile(
    r"(?i)(?:\b(?:kunden?(?:nummer|nr\.?)|customer(?:\s*(?:number|no\.?|id)))\b)"
    r"\s*[:#-]?\s*(?P<value>(?=[A-Z0-9./_-]*\d)[A-Z0-9][A-Z0-9./_-]{3,31})"
)
_DEPOT_RE = re.compile(
    r"(?i)(?:\bdepot(?:\s*[-/]?\s*(?:nummer|nr\.?|kennung|konto))?(?!\w))\s*[:#-]?\s*"
    r"(?P<value>(?=[A-Z0-9./_-]*\d)[A-Z0-9][A-Z0-9./_-]{3,31})"
)
_ACCOUNT_RE = re.compile(
    r"(?i)(?:\b(?:konto(?:nummer|nr\.?)|account(?:\s*(?:number|no\.?|id)))\b)"
    r"\s*[:#-]?\s*(?P<value>(?=[A-Z0-9./_-]*\d)[A-Z0-9][A-Z0-9./_-]{3,31})"
)
_CONTEXT_RE = re.compile(
    r"(?iu)\b(?:kunde|kundin|herr|frau|mitarbeiter(?:in)?|telefon|mobil|mail|e-?mail|"
    r"kontonummer|iban|depot|adresse|wohnort|geburtsdatum)\b"
)
_IDENTIFIER_RE = re.compile(r"(?<!\w)(?=[A-Z0-9./_-]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9][A-Z0-9./_-]{5,31}(?!\w)", re.IGNORECASE)
_DATE_LIKE_RE = re.compile(r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{2,4})$")
_PHONE_CONTEXT_RE = re.compile(r"(?iu)\b(?:telefon|tel\.?|mobil|handy|phone|fax|anrufen|erreichbar)\b")

_SPEAKER_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:\[?[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?(?:[.,][0-9]{1,3})?\]?\s+)?"
    r"(?P<name>[A-ZÄÖÜÀ-ÖØ-Þ][\wÄÖÜäöüßÀ-ÿ.'-]*(?:[ \t]+[A-ZÄÖÜÀ-ÖØ-Þ][\wÄÖÜäöüßÀ-ÿ.'-]*){1,3})"
    r"[ \t]*:(?=\s+\S)"
)
_SPEAKER_TIMESTAMP_RE = re.compile(
    r"(?m)^[ \t]*(?P<name>[A-ZÄÖÜÀ-ÖØ-Þ][\wÄÖÜäöüßÀ-ÿ.'-]*(?:[ _-]+[A-ZÄÖÜÀ-ÖØ-Þ][\wÄÖÜäöüßÀ-ÿ.'-]*){1,3})"
    r"[ \t]+[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?(?:[.,][0-9]{1,3})?[ \t]*$"
)
_PLACEHOLDER_SPEAKER_RE = re.compile(r"(?iu)^(?:speaker|sprecher|person|teilnehmer|unknown|redacted)(?:[_ -]?[a-z]+)?(?:[_ -]?\d+)?$")
_STRUCTURAL_WORDS = {
    "kunde", "kundin", "herr", "frau", "sprecher", "sprecherin", "speaker",
    "moderator", "moderatorin", "teilnehmer", "teilnehmerin", "system", "budget",
    "termin", "depot", "konto", "email", "telefon", "mail", "approval", "threshold",
    "summary", "status", "process", "context", "note", "details", "transcript", "meeting",
    "deutsche", "commerzbank", "volksbank", "sparkasse",
}
_CONTEXT_NEIGHBOUR_CHARS = 1024
_CONTEXT_CURRENT_CHARS = 4096

# Negative *type* evidence, never a blanket allowlist for nearby personal names.
# Whole matched role/system spans are excluded; a name alongside them stays in scope.
_NONPERSON_RE = re.compile(
    r"(?iu)(?<!\w)(?:"
    r"(?:(?:senior|junior|leitende[rns]?|zuständige[rns]?|verantwortliche[rns]?|key|account|relationship|business|risk|credit|product|project|team|it|group|regional|operations|kredit)[ -]+)*"
    r"(?:[\w-]*(?:berater|bearbeiter|mitarbeiter|prüfer|leiter|entwickler|spezialist)(?:in|innen|n)?|manager|analyst|administrator|developer|engineer|owner|lead|scrum master)(?:[_-]\d+)?"
    r"|(?:dax|mdax|sdax|tec-dax)[ -]?(?:konzern|unternehmen)"
    r"|[\w-]*(?:konzern|unternehmen|gesellschaft|abteilung|banksystem|datenbank|plattform|software)"
    r"|(?:front|back|middle)[ -]office|vier[ -]augen[ -]prinzip"
    r"|(?:microsoft[ -]+)?(?:teams|excel|sharepoint|outlook)|microsoft dynamics"
    r"|sap(?:[ -]+(?:s/4hana|hana|erp))?|salesforce|servicenow|jira|confluence"
    r"|deutsche bank|apple pay|crm(?:[ -]system)?|erp(?:[ -]system)?"
    r"|kund(?:e|en|in|innen)|moderator(?:in)?|interviewer(?:in)?"
    r"|(?:speaker|sprecher(?:in)?|teilnehmer(?:in)?)[ _-]*\d+"
    r"|(?:eine?|zwei|drei|mehrere|viele|alle|\d+)\s+personen?"
    r"|[\w-]*(?:vorstand|vorstände|leitungsebene|kompetenzträger|konzernmutter|konzerntochter)"
    r"|(?:chef(?:in|s)?|dezernent(?:en|in|innen)?|kontrolleur(?:e|en|in)?|dolmetsch[\w-]*)"
    r"|(?:kinder(?:n)?|leute(?:n)?|erwachsene[nrms]?|jugendliche[nrms]?|beide[nrms]?)"
    r"|(?:nutzer|benutzer|anwender)(?:in|innen|n)?|vorname[n]?|nachname[n]?"
    r"|(?:[A-Z]\.\s*)?\d+\s+ebene"
    r"|[\w-]*(?:system|werkzeug|programm|projekt|tool|bot)(?:[_-][\w-]+)*"
    r")(?!\w)"
)
_SYSTEM_NAME_RE = re.compile(
    r"(?i:\b(?:system|software|anwendung|plattform|tool)(?:\s+(?:namens|heißt|heisst))?\s+)"
    r"[\"„]?(?P<name>[A-ZÄÖÜ][\w/-]{2,})(?!\w)"
)
_COMMON_NOUN_RE = re.compile(
    r"(?iu)\b(?:im|zum|ein|einem|einen|das|dieses|ist)\s+"
    r"(?P<name>beispiel|muster|winter|sommer|frühling|herbst|keller|braun|weiß|weiss)(?!\w)"
)
_BIRTH_RE = re.compile(r"(?iu)\b(?:geburtsdatum|geburtstag|geburtsjahr|geburt|geboren|geb\.|birth(?:day|date)?|born|dob)(?!\w)")
_CLOCK_RE = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\]?")
_CALENDAR_VALUE_RE = re.compile(
    r"(?iu)(?<!\w)(?:\d{1,4}[./-]\d{1,2}(?:[./-]\d{1,4})?"
    r"|\d{1,2}\.?\s+(?:jan(?:uar|uary)?|feb(?:ruar|ruary)?|märz|maerz|march|apr(?:il)?|mai|may|jun(?:i|e)?|jul(?:i|y)?|aug(?:ust)?|sep(?:tember)?|okt(?:ober)?|oct(?:ober)?|nov(?:ember)?|dez(?:ember)?|dec(?:ember)?)\b"
    r"|(?:18|19|20)\d{2})(?!\w)"
)
_OTHER_DATE_CONTEXT_RE = re.compile(r"(?iu)\b(?:termin|meeting|sitzung|frist|lieferung|appointment|deadline)\b")
_LOGIN_CONTEXT_RE = re.compile(r"(?iu)\b(?:benutzername|username|login|user[- ]?id)\b[^.!?;\n]{0,30}$")
_HUMAN_CONTEXT_RE = re.compile(
    r"(?iu)^\s*(?:fragt|sagt|meint|prüft|antwortet|bestätigt|übernimmt|spricht|meldet|stimmt|erklärt|entscheidet)\b"
)
_AMBIGUOUS_NAMES = {'beispiel', 'muster', 'winter', 'sommer', 'frühling', 'herbst', 'keller', 'braun', 'weiß', 'weiss', 'mai', 'mark', 'will', 'kraft', 'könig'}
_HONORIFIC_RE = re.compile(r'(?iu)\b(?:herrn?|frau|dr\.|prof\.)\s*$')
_DISCOURSE_LINE_RE = re.compile(
    r'(?imu)^[ \t]*(?:(?:hi|hallo|tschüss|tschuess|danke|okay|yup|yeah|no)[ \t.!?,]*)+$'
)
_NAMED_OBJECT_RE = re.compile(
    r'(?iu)\b(?:das(?:\s+(?:system|programm|projekt|tool|verfahren))?|die\s+(?:software|anwendung))'
    r'\s+(?:nennt\s+sich|heißt|heisst)\s+["„]?(?P<name>[\w-]+)'
)
_FIELD_LABEL_RE = re.compile(
    r'(?iu)^(?:[\w-]*(?:nummer|kennung|konto|[ -]id)|(?:account|customer|reference)[ -]?(?:number|id))$'
)
_INITIALS_RE = re.compile(r'(?u)^(?:[A-ZÄÖÜ]\.\s*)+(?:de\.)?$')
_PROCESS_CONTEXT_RE = re.compile(r'(?iu)\b(?:[\w-]*(?:prozess[\w-]*|bearbeitungen)|varianten|abkürzungen)\b')
_NATIONALITY_RE = re.compile(r'(?iu)^(?:schweizer|österreicher|deutsche|franzosen|briten|amerikaner)(?:n|r|s)?$')
_RESIDENTIAL_CONTEXT_RE = re.compile(
    r'(?iu)\b(?:wohnort|wohnadresse|privatadresse|anschrift|postanschrift|wohnhaft|wohnt|wohne|wohnst|'
    r'lebt|lebe|lebst|home\s+address|lives?|resides?|adresse\s*(?:ist|lautet|:))\b'
)
_HANDLE_RE = re.compile(r'(?u)^@?[\w](?:[\w.+-]{0,62}[\w])?$')
_OBFUSCATED_EMAIL_RE = re.compile(r'(?iu)\w+\s*(?:\[at\]|\(at\)|\bat\b)\s*\w+.*(?:\[dot\]|\(dot\)|\bdot\b|\.)\s*\w+')


def _nonperson_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for pattern in (_NONPERSON_RE, _DISCOURSE_LINE_RE) for m in pattern.finditer(text)] + [
        m.span('name') for pattern in (_SYSTEM_NAME_RE, _COMMON_NOUN_RE, _NAMED_OBJECT_RE) for m in pattern.finditer(text)
    ]


def _is_nonperson(text: str, start: int, end: int) -> bool:
    if _HONORIFIC_RE.search(text[max(0, start - 25):start]):
        return False
    # Composite labels can span several adjacent role/system matches. Require
    # every word to be explained; never suppress an adjacent personal name.
    remaining = list(text[start:end])
    for left, right in _nonperson_spans(text):
        for index in range(max(left, start), min(right, end)):
            remaining[index - start] = ' '
    return not any(char.isalnum() for char in remaining)


def _model_evidence(category: str, text: str, start: int, end: int) -> str:
    """Return propose, review or reject using type evidence, not model confidence.

    Impossible types are rejected. Plausible but weakly grounded identifiers stay
    in retained-passage review and cannot seed automatic name propagation.
    """
    value = text[start:end].strip()
    left = text[max(0, start - 120):start]
    human = bool(_HONORIFIC_RE.search(left))
    login = bool(_LOGIN_CONTEXT_RE.search(left))
    if category in {'person', 'username', 'account_id', 'customer_id', 'depot_id'}:
        if _is_nonperson(text, start, end) and not (category == 'username' and login):
            return 'reject'
    if category == 'email':
        if _EMAIL_RE.fullmatch(value) or _OBFUSCATED_EMAIL_RE.fullmatch(value):
            return 'propose'
        return 'review' if '@' in value else 'reject'
    if category == 'iban':
        if _iban_valid(value):
            return 'propose'
        compact = re.sub(r'\s|-', '', value)
        # A transcription error in a checksum does not make an account public.
        return 'propose' if re.fullmatch(r'(?i)[A-Z]{2}\d{2}[A-Z0-9]{11,30}', compact) else 'reject'
    if category in {'account_id', 'customer_id', 'depot_id'}:
        if _FIELD_LABEL_RE.fullmatch(value):
            return 'reject'
        if any(char.isdigit() for char in value) and re.fullmatch(r'[\w ./-]{3,64}', value):
            return 'propose'
        return 'review' if _HANDLE_RE.fullmatch(value) else 'reject'
    if category == 'username':
        if not _HANDLE_RE.fullmatch(value):
            return 'reject'
        if login or value.startswith('@') or any(char.isdigit() for char in value) or '_' in value or '.' in value:
            return 'propose'
        return 'review'
    if category == 'person' and not human:
        if len(value.rstrip('.')) == 1:
            return 'review'
        if _INITIALS_RE.fullmatch(value) and _PROCESS_CONTEXT_RE.search(left):
            return 'reject'
        if _NATIONALITY_RE.fullmatch(value) and re.match(r'(?iu)\s+in\s+(?:der\s+)?\w+', text[end:]):
            return 'reject'
    if category == 'address':
        local_context = re.split(r'[.!?;\n]', left)[-1]
        if any(char.isdigit() for char in value) or _RESIDENTIAL_CONTEXT_RE.search(local_context + value):
            return 'propose'
        return 'review'
    return 'propose'


def _birth_context(text: str, start: int, end: int) -> bool:
    value = text[start:end].strip()
    if _CLOCK_RE.fullmatch(value) or not _CALENDAR_VALUE_RE.search(value):
        return False
    # Keep cues in the same sentence/field, including a cue on the previous line.
    left = re.split(r'[!?;,]|(?<!geb)\.(?:\s|$)', text[max(0, start - 100):start], flags=re.IGNORECASE)[-1]
    right = re.split(r'[!?;]|\.(?:\s|$)', text[end:end + 60])[0]
    # A later appointment in a birth-related sentence is not itself a birth date.
    if _OTHER_DATE_CONTEXT_RE.search(left) and not _BIRTH_RE.search(left + ' ' + value):
        return False
    return bool(_BIRTH_RE.search(left + ' ' + value + ' ' + right))


def _name_form(value: str) -> str:
    return re.sub(r'[ _\t\u00a0]+', ' ', value.strip()).casefold()


def _name_pattern(value: str) -> str:
    return r'(?<!\w)' + r'[ _\t\u00a0]+'.join(re.escape(part) for part in re.split(r'[ _\t\u00a0]+', value.strip())) + r'(?!\w)'


def _error(message: str) -> AnonymizerError:
    """Create a safe error without interpolating policy or source values."""

    return AnonymizerError(message)


def _normalise_aliases(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    entries: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        iterable = [{"entity_key": key, "aliases": value} for key, value in raw.items()]
    elif isinstance(raw, list):
        iterable = raw
    else:
        raise _error("Policy aliases must be a list or mapping")
    for item in iterable:
        if not isinstance(item, dict) or not isinstance(item.get("entity_key"), str):
            raise _error("Each policy alias entry needs an entity_key")
        if set(item) - {"entity_key", "aliases", "values", "names"}:
            raise _error("Policy alias entry contains an unknown field")
        values = item.get("aliases", item.get("values", item.get("names")))
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
            raise _error("Each policy alias entry needs non-empty string aliases")
        entries.append({"entity_key": item["entity_key"], "aliases": list(dict.fromkeys(values))})
    return entries


def _reject_unknown_policy_keys(raw: dict[str, Any]) -> None:
    unknown = set(raw) - _POLICY_KEYS
    if unknown:
        raise _error("Detection policy contains an unknown field")
    if isinstance(raw.get("model"), dict) and set(raw["model"]) - _MODEL_POLICY_KEYS:
        raise _error("Detection policy model contains an unknown field")
    if isinstance(raw.get("rules"), dict) and set(raw["rules"]) - _RULE_POLICY_KEYS:
        raise _error("Detection policy rules contain an unknown field")


def load_policy(path: Path | None) -> dict[str, Any]:
    """Load and validate a JSON policy, or return an independent default policy.

    A policy is intentionally JSON-only: this keeps the offline package small and
    avoids accepting executable or surprising configuration formats.  The normalised
    result is JSON-compatible and safe to persist with a run record.
    """

    if path is None:
        return copy.deepcopy(_DEFAULT_POLICY)
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error("Could not read a valid JSON detection policy") from exc
    if not isinstance(raw, dict):
        raise _error("Detection policy must be a JSON object")
    _reject_unknown_policy_keys(raw)
    if raw.get("schema_version") != 1:
        raise _error("Detection policy schema_version must be 1")

    policy = copy.deepcopy(_DEFAULT_POLICY)
    policy.update({key: value for key, value in raw.items() if key not in {"model", "rules", "aliases"}})
    if "model" in raw:
        if not isinstance(raw["model"], dict):
            raise _error("Policy model must be an object")
        policy["model"].update(raw["model"])
    if "rules" in raw:
        if not isinstance(raw["rules"], dict):
            raise _error("Policy rules must be an object")
        policy["rules"].update(raw["rules"])
    policy["aliases"] = _normalise_aliases(raw.get("aliases", []))

    _normalise_thresholds(policy["model"], raw.get("model") if isinstance(raw.get("model"), dict) else None)

    labels = policy["model"].get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label for label in labels):
        raise _error("Policy model.labels must be a non-empty list of strings")
    # Labels outside this map are deliberately rejected: accepting arbitrary model
    # labels would turn business entities into PII under the default policy.
    if any(label not in _MODEL_CATEGORY for label in labels):
        raise _error("Policy model.labels contains a non-PII or unsupported label")
    _validate_thresholds(policy["model"])
    for key in ("chunk_tokens", "chunk_overlap_tokens"):
        value = policy["model"].get(key)
        if not isinstance(value, int) or value <= 0:
            raise _error(f"Policy model.{key} must be a positive integer")
    if policy["model"]["chunk_overlap_tokens"] >= policy["model"]["chunk_tokens"]:
        raise _error("Policy model chunk overlap must be smaller than chunk size")
    for key, value in policy["rules"].items():
        if not isinstance(value, bool):
            raise _error("Policy rules values must be booleans")
    return policy


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalise_thresholds(model: dict[str, Any], raw_model: dict[str, Any] | None = None) -> None:
    """Normalise compatibility and per-label threshold spellings in-place."""

    raw_model = {} if raw_model is None else raw_model
    has_new = any(key in raw_model for key in ("proposal_threshold", "review_threshold", "thresholds", "label_thresholds"))
    threshold = model.get("threshold", 0.35)
    # Explicit new fields take precedence over the legacy spelling. A policy
    # using only the old field retains its previous single-threshold meaning.
    if "threshold" in raw_model and not has_new and _is_number(threshold):
        model["proposal_threshold"] = threshold
        model["review_threshold"] = threshold
    else:
        model.setdefault("proposal_threshold", threshold)
        model.setdefault("review_threshold", 0.20)

    supplied = raw_model.get("thresholds", raw_model.get("label_thresholds", model.get("thresholds", {})))
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise _error("Policy model.thresholds must be an object")
    normalised: dict[str, dict[str, float]] = {}
    for label, config in supplied.items():
        if not isinstance(label, str) or label not in _MODEL_CATEGORY or label not in model["labels"]:
            raise _error("Policy model.thresholds contains an unsupported label")
        if _is_number(config):
            proposal, review = float(config), float(model["review_threshold"])
        elif isinstance(config, dict):
            allowed = {"proposal", "review", "proposal_threshold", "review_threshold"}
            if set(config) - allowed:
                raise _error("Policy model threshold entry contains an unknown field")
            proposal = config.get("proposal", config.get("proposal_threshold", model["proposal_threshold"]))
            review = config.get("review", config.get("review_threshold", model["review_threshold"]))
        else:
            raise _error("Policy model threshold entries must be numbers or objects")
        if not _is_number(proposal) or not _is_number(review):
            raise _error("Policy model thresholds must be numbers")
        normalised[label] = {"proposal": float(proposal), "review": float(review)}
    model["thresholds"] = normalised


def _validate_thresholds(model: dict[str, Any]) -> None:
    for key in ("threshold", "proposal_threshold", "review_threshold"):
        if not _is_number(model.get(key)) or not 0 <= model[key] <= 1:
            raise _error(f"Policy model.{key} must be between 0 and 1")
    if model["review_threshold"] > model["proposal_threshold"]:
        raise _error("Policy model.review_threshold cannot exceed proposal_threshold")
    for label, config in model.get("thresholds", {}).items():
        if not 0 <= config["review"] <= config["proposal"] <= 1:
            raise _error(f"Policy model threshold for {label} is invalid")


def _finding(
    segment_id: str,
    text: str,
    start: int,
    end: int,
    category: str,
    score: float,
    detector: str,
    entity_key: str | None = None,
    review_only: bool = False,
) -> dict[str, Any]:
    if not 0 <= start < end <= len(text):
        raise _error("Detector produced an out-of-bounds span")
    result: dict[str, Any] = {
        "id": f"{segment_id}:{start}:{end}:{category}:{detector}",
        "segment_id": segment_id,
        "start": start,
        "end": end,
        "text": text[start:end],
        "category": category,
        "score": round(float(score), 6),
        "detector": detector,
    }
    if entity_key is not None:
        result["entity_key"] = entity_key
    if review_only:
        result["review_only"] = True
    return result


def _iban_valid(value: str) -> bool:
    compact = re.sub(r"[ -]", "", value).upper()
    if len(compact) < 15 or len(compact) > 34 or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
    remainder = 0
    for char in numeric:
        remainder = (remainder * 10 + int(char)) % 97
    return remainder == 1


def _is_phone_candidate(text: str, match: re.Match[str]) -> bool:
    value = match.group().strip()
    digits = re.sub(r"\D", "", value)
    if len(digits) < 7 or _DATE_LIKE_RE.fullmatch(value):
        return False
    if value.startswith("+"):
        return True
    # A grouped German-looking number is sufficiently distinctive. Plain large
    # business figures (volume, amount, timestamp) are intentionally excluded.
    if value.startswith("0") and bool(re.search(r"[ .()/ -]", value)):
        return True
    left = max(0, match.start() - 32)
    right = min(len(text), match.end() + 32)
    return _PHONE_CONTEXT_RE.search(text[left:right]) is not None


def _asset_manifest() -> tuple[dict[str, Any], str]:
    """Read the packaged model manifest and return it with a stable fingerprint."""

    try:
        resource = importlib.resources.files("transcript_anonymizer").joinpath("model-assets.json")
        raw = resource.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise _error("Packaged model asset manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise _error("Packaged model asset manifest is unsupported")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise _error("Packaged model asset manifest has no files")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise _error("Packaged model asset manifest has an invalid file")
        relative = Path(item["path"])
        sha256 = item.get("sha256")
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise _error("Packaged model asset manifest has an unsafe path")
        if not isinstance(item.get("size"), int) or item["size"] < 0 or not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise _error("Packaged model asset manifest has an invalid checksum")
        seen.add(relative.as_posix())
    return manifest, hashlib.sha256(raw).hexdigest()


def _verify_model_assets(model_path: Path) -> None:
    """Verify every local checkpoint file before importing/loading model weights."""

    manifest, _ = _asset_manifest()
    root = model_path.resolve()
    expected = {item["path"] for item in manifest["files"]}
    actual: set[str] = set()
    try:
        for path in model_path.rglob("*"):
            if path.is_file():
                resolved = path.resolve()
                if root not in resolved.parents:
                    raise _error("Model asset path escapes its local directory")
                actual.add(path.relative_to(model_path).as_posix())
    except OSError as exc:
        raise _error("Could not inspect local model assets") from exc
    if actual != expected:
        raise _error("Local model assets do not match the pinned manifest")
    for item in manifest["files"]:
        path = model_path / item["path"]
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    size += len(block)
                    digest.update(block)
        except OSError as exc:
            raise _error("Could not read a local model asset") from exc
        if size != item["size"] or digest.hexdigest() != item["sha256"]:
            raise _error("Local model assets failed pinned checksum verification")


def _assemble_model_parts(model_path: Path) -> None:
    """Rebuild a split release's checkpoint before the normal pinned check."""

    manifest_path = model_path.parent / "model-parts.json"
    if not manifest_path.is_file():
        return
    try:
        plan = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = plan["target"]
        parts = plan["parts"]
        expected_size = plan["bytes"]
        expected_sha256 = plan["sha256"]
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error("Split model package manifest is invalid") from exc
    if (
        not isinstance(target, str)
        or target != "model.safetensors"
        or not isinstance(parts, list)
        or not parts
        or type(expected_size) is not int
        or expected_size < 0
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise _error("Split model package manifest is invalid")
    destination = model_path / target
    if destination.is_file():
        return
    parts_root = model_path.parent / "model-parts"
    # v0.2.4 placed parts beside `_internal`; retain that verified-only layout
    # as a compatibility fallback so already-downloaded releases remain usable.
    if not parts_root.is_dir():
        legacy_parts_root = model_path.parent.parent / "model-parts"
        if legacy_parts_root.is_dir():
            parts_root = legacy_parts_root
    temporary = destination.with_suffix(destination.suffix + ".assembling")
    digest, size = hashlib.sha256(), 0
    try:
        with temporary.open("xb") as output:
            for index, item in enumerate(parts, 1):
                if not isinstance(item, dict) or item.get("name") != f"model.safetensors.part{index:03d}":
                    raise _error("Split model package manifest is invalid")
                part = parts_root / item["name"]
                if not part.is_file() or type(item.get("bytes")) is not int or not isinstance(item.get("sha256"), str):
                    raise _error("Extract every model asset package into this application folder")
                part_digest, part_size = hashlib.sha256(), 0
                with part.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        part_size += len(chunk)
                        part_digest.update(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        output.write(chunk)
                if part_size != item["bytes"] or part_digest.hexdigest() != item["sha256"]:
                    raise _error("A split model asset package is incomplete or modified")
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise _error("Reassembled model asset failed pinned checksum verification")
        temporary.replace(destination)
        shutil.rmtree(parts_root)
        manifest_path.unlink()
    except AnonymizerError:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise _error("Could not assemble the local model assets") from exc


class Detector:
    """Run deterministic rules and, when explicitly configured, a local GLiNER2 model."""

    def __init__(self, policy: dict[str, Any], model_path: Path | None = None, rules_only: bool = False):
        self.policy = load_policy(None) if policy is None else _validate_in_memory_policy(policy)
        self.rules_only = rules_only
        self.model_path = Path(model_path) if model_path is not None else None
        self._model: Any | None = None
        self.progress = None
        _, manifest_sha256 = _asset_manifest()
        self.metadata: dict[str, Any] = {
            "schema_version": 1,
            "metadata_version": 2,
            "detector_version": "quality-v4",
            "policy_version": self.policy.get("schema_version", 1),
            "rules": "builtin-v2",
            "rules_only": rules_only,
            "model": None,
            "limitations": [
                "Candidates and scores are not calibrated probabilities",
                "Indirect personal identification still requires ranked human review",
                "Overlapping candidates are preserved for the parent resolver",
            ],
        }
        if rules_only:
            self.metadata["model"] = {
                "status": "disabled",
                "reason": "explicit_rules_only",
                "name": self.policy["model"].get("name", MODEL_NAME),
                "revision": self.policy["model"].get("revision", MODEL_REVISION),
                "manifest_sha256": manifest_sha256,
            }
        else:
            self.metadata["model"] = {
                # Keep metadata stable before and after lazy loading.  Workflow
                # revisions compare this object when reviewing corrections.
                "status": "configured",
                "name": self.policy["model"].get("name", MODEL_NAME),
                "revision": self.policy["model"].get("revision", MODEL_REVISION),
                "path": str(self.model_path) if self.model_path is not None else None,
                "labels": list(self.policy["model"]["labels"]),
                "proposal_threshold": self.policy["model"]["proposal_threshold"],
                "review_threshold": self.policy["model"]["review_threshold"],
                "thresholds": copy.deepcopy(self.policy["model"].get("thresholds", {})),
                "manifest_sha256": manifest_sha256,
            }

    @staticmethod
    def _load_model(model_path: Path) -> Any:
        if not model_path.is_dir():
            raise _error("Configured local model assets are missing")
        _assemble_model_parts(model_path)
        _verify_model_assets(model_path)
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        # GLiNER prints a Unicode banner unconditionally. It fails with legacy
        # Windows output encodings; GUI executables may have no streams at all.
        # Serialize startup redirection even for callers outside the single-worker
        # desktop, so overlapping contexts cannot restore the wrong streams.
        with _MODEL_CONSOLE_LOCK, redirect_stdout(_MODEL_CONSOLE), redirect_stderr(_MODEL_CONSOLE):
            try:
                from gliner2 import AutoExtractor  # type: ignore
            except (ImportError, ModuleNotFoundError) as exc:
                raise _error("gliner2[local] is required for model detection") from exc
            try:
                import torch  # type: ignore

                torch.set_num_threads(2)
                try:
                    torch.set_num_interop_threads(2)
                except RuntimeError:
                    # A caller may have initialized the inter-op pool already.
                    pass
                return AutoExtractor.from_pretrained(
                    str(model_path), local_files_only=True, map_location="cpu"
                )
            except Exception as exc:
                raise _error("Configured local model could not be loaded") from exc

    def detect(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        checked_segments = [_segment_text(segment) for segment in segments]
        findings: list[dict[str, Any]] = []
        speaker_findings = self._speaker_findings(checked_segments)
        findings.extend(speaker_findings)
        for index, (segment_id, text) in enumerate(checked_segments, 1):
            findings.extend(self._rule_findings(segment_id, text))
            if not self.rules_only:
                findings.extend(self._model_findings(segment_id, text))
                # A bounded neighbouring view gives the model transcript
                # continuity.  Findings are mapped back to exact source spans;
                # the ordinary segment pass above still covers omitted tails.
                if len(checked_segments) > 1 and index % 2 == 1 and len(text) <= _CONTEXT_CURRENT_CHARS:
                    findings.extend(self._model_context_findings(checked_segments, index - 1))
            if self.progress is not None:
                self.progress(index, len(segments))
        # A model may rediscover a structural full name with a different score.
        # Keep it in the same explicit document-scoped group rather than allowing
        # score ordering to split the identity from the speaker evidence.
        known_names = {
            item["text"].casefold(): item["entity_key"]
            for item in speaker_findings
            if item["detector"] == "rule:speaker" and isinstance(item.get("entity_key"), str)
        }
        known_names.update(
            {
                alias.casefold(): entry["entity_key"]
                for entry in self.policy["aliases"]
                for alias in entry["aliases"]
            }
        )
        for finding in findings:
            if finding["category"] == "person" and finding["detector"].startswith("model:"):
                entity_key = known_names.get(finding["text"].casefold())
                if entity_key is not None:
                    finding["entity_key"] = entity_key
        findings.extend(self._document_aliases(checked_segments, findings))
        # Context windows overlap by design.  Collapse only exact duplicates
        # produced by those windows; preserve distinct detectors and overlapping
        # spans for the parent resolver/ranker.
        unique: dict[tuple[Any, ...], dict[str, Any]] = {}
        for finding in findings:
            key = (finding["segment_id"], finding["start"], finding["end"], finding["category"], finding["detector"], finding.get("entity_key"))
            previous = unique.get(key)
            if previous is None or finding["score"] > previous["score"]:
                unique[key] = finding
        findings = list(unique.values())
        findings.sort(key=lambda item: (item["segment_id"], item["start"], item["end"], item["category"], item["detector"], item.get("entity_key", "")))
        return findings

    def _speaker_findings(self, segments: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Extract conservative ``Name:`` turn labels and propagate full names.

        The generated key is stable for a normalized name within a run, so
        corrections/rescans do not silently split an existing replacement group.
        Only full names propagate automatically.  A surname-only suggestion is
        retained for review and never receives the speaker entity key.
        """

        structural: list[tuple[str, str, int, int, str]] = []
        for segment_id, text in segments:
            matches = list(_SPEAKER_LINE_RE.finditer(text)) + list(_SPEAKER_TIMESTAMP_RE.finditer(text))
            seen_spans: set[tuple[int, int]] = set()
            for match in matches:
                span = (match.start("name"), match.end("name"))
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                name = match.group("name").strip()
                if any(left < match.end('name') and match.start('name') < right
                       for left, right in _nonperson_spans(text)):
                    continue
                words = name.split()
                components = re.split(r"[ _-]+", name)
                if (
                    len(words) < 2 and len(components) < 2
                ) or any(word.casefold().rstrip(".") in _STRUCTURAL_WORDS for word in components) or _PLACEHOLDER_SPEAKER_RE.fullmatch(name):
                    continue
                structural.append((segment_id, text, match.start("name"), match.end("name"), name))
        if not structural:
            return []
        explicit = {
            alias.casefold(): entry["entity_key"]
            for entry in self.policy["aliases"]
            for alias in entry["aliases"]
            if len(alias.split()) >= 2
        }
        names: dict[str, tuple[str, str]] = {}
        for _, _, _, _, name in structural:
            folded = name.casefold()
            names.setdefault(folded, (name, explicit.get(folded, f"speaker:{hashlib.sha256(folded.encode('utf-8')).hexdigest()[:16]}")))

        output: list[dict[str, Any]] = []
        structural_spans = {(sid, start, end) for sid, _, start, end, _ in structural}
        for folded, (name, entity_key) in names.items():
            for segment_id, text in segments:
                for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE):
                    span_key = (segment_id, match.start(), match.end())
                    is_structural = span_key in structural_spans
                    output.append(_finding(
                        segment_id, text, match.start(), match.end(), "person", 0.97 if is_structural else 0.78,
                        "rule:speaker" if is_structural else "rule:speaker-propagation", entity_key,
                    ))
                # Surname-only mentions are useful evidence but are too
                # ambiguous to merge with the speaker automatically.
                surname = name.split()[-1]
                if len(surname) >= 3:
                    full_spans = [m.span() for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE)]
                    for match in re.finditer(r"(?<!\w)" + re.escape(surname) + r"(?!\w)", text, re.IGNORECASE):
                        if any(start <= match.start() and match.end() <= end for start, end in full_spans):
                            continue
                        output.append(_finding(segment_id, text, match.start(), match.end(), "person", 0.58, "rule:speaker-alias-suggestion", review_only=True))
        return output

    def _document_aliases(self, segments, findings):
        """Link local, evidenced variants; ambiguity never chooses a person."""
        anchors = {}
        explicit = {_name_form(alias): entry['entity_key'] for entry in self.policy['aliases'] for alias in entry['aliases']}
        for finding in findings:
            if finding['detector'] == 'rule:speaker':
                anchors[_name_form(finding['text'])] = (finding['text'], finding['entity_key'])
        for finding in findings:
            name = finding['text']
            form = _name_form(name)
            if finding['category'] == 'person' and not finding.get('review_only') and finding['score'] >= 0.8 and len(form.split()) >= 2:
                anchors.setdefault(form, (name, explicit.get(form, finding.get('entity_key') or f"name:{hashlib.sha256(form.encode()).hexdigest()[:16]}")))
        for entry in self.policy['aliases']:
            for name in entry['aliases']:
                if len(_name_form(name).split()) >= 2:
                    anchors[_name_form(name)] = (name, entry['entity_key'])
        variants = {}

        def add(name, key):
            variants.setdefault(_name_form(name), (name, set()))[1].add(key)

        for full, key in anchors.values():
            plain = re.sub(r'(?i)^(?:(?:dr|prof)\.\s*)+', '', full).replace('_', ' ')
            parts = plain.split()
            if len(parts) < 2:
                continue
            for value in (full, plain, parts[0], parts[-1], parts[0][0] + '. ' + parts[-1]):
                add(value, key)
            for _, text in segments:
                pattern = _name_pattern(plain) + r'\s*,?\s*(?i:genannt|alias)\s+(?P<nick>[A-ZÄÖÜ][\w-]{1,30})(?!\w)'
                for match in re.finditer(pattern, text):
                    if not _is_nonperson(text, *match.span('nick')):
                        add(match.group('nick'), key)
        for entry in self.policy['aliases']:
            for name in entry['aliases']:
                variants[_name_form(name)] = (name, {entry['entity_key']})
        output = []
        for form, (name, keys) in variants.items():
            pattern = _name_pattern(name)
            if re.match(r'^\w\. ', name):
                pattern = r'(?<!\w)' + re.escape(name[0]) + r'\.?[ \t\u00a0]*' + re.escape(name[3:]) + r'(?!\w)'
            for sid, text in segments:
                for match in re.finditer(pattern, text, re.IGNORECASE if len(form.split()) > 1 or form in explicit else 0):
                    if _is_nonperson(text, *match.span()) and form not in explicit:
                        continue
                    # Partial components inside a full name are already covered.
                    if any(f['segment_id'] == sid and f['start'] <= match.start() and match.end() <= f['end'] and f.get('entity_key') in keys and len(_name_form(f['text']).split()) >= 2 for f in findings):
                        continue
                    human = bool(re.search(r'(?iu)\b(?:herrn?|frau|dr\.)\s*$', text[max(0, match.start() - 20):match.start()]) or _HUMAN_CONTEXT_RE.search(text[match.end():]))
                    if (form in _AMBIGUOUS_NAMES or len(keys) > 1) and not human and form not in explicit:
                        continue
                    key = next(iter(keys)) if len(keys) == 1 else None
                    output.append(_finding(sid, text, match.start(), match.end(), 'person', 0.84, 'rule:document-alias', key))
                    for finding in findings:
                        if finding['category'] == 'person' and finding['segment_id'] == sid and finding['start'] == match.start() and finding['end'] == match.end() and key is not None and not finding.get('review_only'):
                            finding['entity_key'] = key
        return output

    def _rule_findings(self, segment_id: str, text: str) -> list[dict[str, Any]]:
        rules = self.policy["rules"]
        output: list[dict[str, Any]] = []
        if rules.get("email", True):
            output.extend(_finding(segment_id, text, m.start(), m.end(), "email", 0.99, "rule:email") for m in _EMAIL_RE.finditer(text))
        if rules.get("phone", True):
            for m in _PHONE_RE.finditer(text):
                if _is_phone_candidate(text, m):
                    output.append(_finding(segment_id, text, m.start(), m.end(), "phone", 0.94, "rule:phone"))
        if rules.get("iban", True):
            for m in _IBAN_RE.finditer(text):
                if _iban_valid(m.group()):
                    output.append(_finding(segment_id, text, m.start(), m.end(), "iban", 1.0, "rule:iban"))
        for key, regex, category, score in (
            ("customer_id", _CUSTOMER_RE, "customer_id", 0.86),
            ("depot_id", _DEPOT_RE, "depot_id", 0.86),
            ("account_id", _ACCOUNT_RE, "account_id", 0.84),
        ):
            if rules.get(key, True):
                for match in regex.finditer(text):
                    start, end = match.span("value")
                    while end > start and text[end - 1] == ".":
                        end -= 1
                    output.append(_finding(segment_id, text, start, end, category, score, f"rule:{key}"))
        for alias in self.policy["aliases"]:
            for value in alias["aliases"]:
                pattern = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.IGNORECASE)
                for match in pattern.finditer(text):
                    output.append(_finding(segment_id, text, match.start(), match.end(), "person", 0.98, "rule:alias", alias["entity_key"]))
        return output

    def _model_findings(self, segment_id: str, text: str) -> list[dict[str, Any]]:
        self._ensure_model()
        if self._model is None:  # defensive: constructor never silently enables rules-only
            raise _error("Model detector is not available")
        findings: list[dict[str, Any]] = []
        for chunk_start, chunk_end in self._chunks(text):
            chunk = text[chunk_start:chunk_end]
            result = self._extract(chunk)
            findings.extend(self._parse_model_result(segment_id, text, chunk_start, chunk, result))
        unique: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in findings:
            key = (item["start"], item["end"], item["category"], item["detector"], item.get("entity_key"))
            if key not in unique or item["score"] > unique[key]["score"]:
                unique[key] = item
        return list(unique.values())

    def _threshold_for(self, label: str, kind: str) -> float:
        config = self.policy["model"].get("thresholds", {}).get(label)
        if config is not None:
            return float(config[kind])
        return float(self.policy["model"][f"{kind}_threshold"])

    def _context_window(self, segments: list[tuple[str, str]], index: int) -> tuple[str, list[tuple[str, int, int, int, int]]]:
        """Return a bounded context string and its source mapping pieces."""

        pieces: list[tuple[str, int, int, int, int]] = []
        chunks: list[str] = []
        cursor = 0

        def add(segment_id: str, source: str, start: int, end: int) -> None:
            nonlocal cursor
            if end <= start:
                return
            if chunks:
                chunks.append("\n")
                cursor += 1
            chunks.append(source[start:end])
            pieces.append((segment_id, start, end, cursor, cursor + end - start))
            cursor += end - start

        if index > 0:
            previous_id, previous = segments[index - 1]
            add(previous_id, previous, max(0, len(previous) - _CONTEXT_NEIGHBOUR_CHARS), len(previous))
        current_id, current = segments[index]
        if len(current) <= _CONTEXT_CURRENT_CHARS:
            add(current_id, current, 0, len(current))
        else:
            half = _CONTEXT_CURRENT_CHARS // 2
            add(current_id, current, 0, half)
            add(current_id, current, len(current) - half, len(current))
        if index + 1 < len(segments):
            following_id, following = segments[index + 1]
            add(following_id, following, 0, min(len(following), _CONTEXT_NEIGHBOUR_CHARS))
        return "".join(chunks), pieces

    @staticmethod
    def _map_context_span(start: int, end: int, pieces: list[tuple[str, int, int, int, int]]) -> list[tuple[str, int, int]]:
        mapped: list[tuple[str, int, int]] = []
        for segment_id, source_start, source_end, context_start, context_end in pieces:
            overlap_start = max(start, context_start)
            overlap_end = min(end, context_end)
            if overlap_start < overlap_end:
                mapped.append((segment_id, source_start + overlap_start - context_start, source_start + overlap_end - context_start))
        if not mapped:
            raise _error("Local model returned a span outside source context")
        return mapped

    def _model_context_findings(self, segments: list[tuple[str, str]], index: int) -> list[dict[str, Any]]:
        context, pieces = self._context_window(segments, index)
        if not context:
            return []
        source_by_id = {segment_id: text for segment_id, text in segments}
        output: list[dict[str, Any]] = []
        for chunk_start, chunk_end in self._chunks(context):
            chunk = context[chunk_start:chunk_end]
            result = self._extract(chunk)
            # Parse against the context first, then split any cross-segment span
            # into representable source occurrences.
            parsed = self._parse_model_result("__context__", context, chunk_start, chunk, result)
            for finding in parsed:
                for segment_id, start, end in self._map_context_span(finding["start"], finding["end"], pieces):
                    output.append(_finding(
                        segment_id, source_by_id[segment_id], start, end, finding["category"], finding["score"],
                        finding["detector"], finding.get("entity_key"), bool(finding.get("review_only")),
                    ))
        return output

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None:
            raise _error("A local model_path is required unless rules_only=True")
        self._model = self._load_model(self.model_path)

    def _chunks(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            processor = getattr(self._model, "processor", None)
            tokenizer = getattr(processor, "tokenizer", None)
        budget = self.policy["model"]["chunk_tokens"]
        if tokenizer is None:
            # A mock or third-party adapter may not expose a tokenizer.  Keep this
            # bounded and deterministic; real GLiNER2 always has one.
            size = max(1, min(budget * 4, 4096))
            overlap = min(self.policy["model"]["chunk_overlap_tokens"] * 4, size - 1)
            return _char_chunks(len(text), size, overlap)
        actual = getattr(tokenizer, "model_max_length", None)
        encoder = getattr(self._model, "encoder", None)
        encoder_config = getattr(encoder, "config", None)
        encoder_limit = getattr(encoder_config, "max_position_embeddings", None)
        if isinstance(encoder_limit, int) and encoder_limit > 0:
            if not isinstance(actual, int) or actual >= 10_000_000:
                actual = encoder_limit
            else:
                actual = min(actual, encoder_limit)
        if isinstance(actual, int) and 0 < actual < 10_000_000:
            # GLiNER2 appends schema markers and label tokens to text before
            # encoding. Reserve those tokens so the encoder cannot truncate the
            # nominal text chunk at its positional limit.
            budget = min(budget, max(1, actual - self._schema_token_reserve(tokenizer) - 8))
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            offsets = encoded["offset_mapping"]
        except Exception as exc:
            raise _error("Configured model tokenizer could not tokenize input") from exc
        if not offsets:
            return []
        offsets = [(int(pair[0]), int(pair[1])) for pair in offsets if pair[1] > pair[0]]
        if not offsets:
            return []
        budget = max(1, budget)
        overlap = min(self.policy["model"]["chunk_overlap_tokens"], budget - 1)
        step = max(1, budget - overlap)
        chunks: list[tuple[int, int]] = []
        start_index = 0
        while start_index < len(offsets):
            end_index = min(start_index + budget, len(offsets))
            start, end = offsets[start_index][0], offsets[end_index - 1][1]
            if chunks and start <= chunks[-1][0] and end <= chunks[-1][1]:
                break
            chunks.append((start, min(len(text), end)))
            if end_index == len(offsets):
                break
            start_index += step
        if chunks[-1][1] < len(text):
            tail_start = offsets[max(0, len(offsets) - budget)][0]
            chunks.append((tail_start, len(text)))
        return chunks

    def _schema_token_reserve(self, tokenizer: Any) -> int:
        """Conservatively reserve GLiNER schema/prompt tokens for one chunk."""

        token_count = 0
        measured = False
        for label in self.policy["model"]["labels"]:
            ids = None
            try:
                encoded = tokenizer(label, add_special_tokens=False, truncation=False)
                ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            if isinstance(ids, list):
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                token_count += len(ids)
                measured = True
        if measured:
            return max(32, token_count + 4 * len(self.policy["model"]["labels"]) + 16)
        # Test doubles and third-party tokenizers may expose only offset mapping.
        return 64

    def _extract(self, chunk: str) -> Any:
        labels = self.policy["model"]["labels"]
        # The model is asked for the broadest review pass.  Per-label filtering
        # happens after parsing, so a low-confidence candidate is not lost merely
        # because another label has a higher proposal threshold.
        threshold = min(self._threshold_for(label, "review") for label in labels)
        try:
            return self._model.extract_entities(chunk, labels, threshold=threshold, include_confidence=True, include_spans=True)
        except TypeError as exc:
            # A mock adapter or older local release may lack optional output flags;
            # retry only for that API mismatch, never for arbitrary model failures.
            message = str(exc)
            if not any(token in message for token in ("unexpected keyword", "keyword argument", "positional argument")):
                raise _error("Local model inference failed") from exc
            try:
                return self._model.extract_entities(chunk, labels, threshold=threshold)
            except TypeError as inner:
                # Never retry without a threshold: that would silently turn a
                # bounded review pass into an unbounded model result.
                raise _error("Local model does not support thresholded inference") from inner
            except Exception as inner:
                raise _error("Local model inference failed") from inner
        except Exception as exc:
            raise _error("Local model inference failed") from exc

    def _parse_model_result(self, segment_id: str, source: str, chunk_start: int, chunk: str, result: Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            raise _error("Local model returned an invalid result")
        entities = result.get("entities", result)
        rows: list[tuple[str, Any]] = []
        if isinstance(entities, dict):
            for label, values in entities.items():
                if label in _MODEL_CATEGORY:
                    if isinstance(values, list):
                        rows.extend((label, value) for value in values)
                    else:
                        rows.append((label, values))
        elif isinstance(entities, list):
            for value in entities:
                if isinstance(value, dict):
                    label = value.get("label", value.get("type", value.get("entity_type")))
                    if isinstance(label, str) and label in _MODEL_CATEGORY:
                        rows.append((label, value))
        else:
            raise _error("Local model returned an invalid entity collection")

        output: list[dict[str, Any]] = []
        for label, value in rows:
            category = _MODEL_CATEGORY[label]
            detector = f"model:gliner2:{label}"
            score = 0.5
            local_start: int | None = None
            local_end: int | None = None
            entity_key = None
            if isinstance(value, dict):
                raw_text = value.get("text", value.get("span", value.get("value")))
                local_start = _integer_or_none(value.get("start", value.get("start_idx")))
                local_end = _integer_or_none(value.get("end", value.get("end_idx")))
                score = value.get("score", value.get("confidence", score))
                entity_key = value.get("entity_key") if isinstance(value.get("entity_key"), str) else None
            else:
                raw_text = value
            if local_start is None or local_end is None:
                raise _error("Local model returned an entity without exact offsets")
            else:
                spans = [(local_start, local_end)]
            try:
                score = float(score)
            except (TypeError, ValueError) as exc:
                raise _error("Local model returned an invalid score") from exc
            if not 0 <= score <= 1:
                raise _error("Local model returned an out-of-range score")
            review_threshold = self._threshold_for(label, "review")
            proposal_threshold = self._threshold_for(label, "proposal")
            if score < review_threshold:
                continue
            for local_start, local_end in spans:
                if local_start < 0 or local_end <= local_start or local_end > len(chunk):
                    raise _error("Local model returned an out-of-bounds span")
                if isinstance(raw_text, str) and raw_text and chunk[local_start:local_end] != raw_text:
                    raise _error("Local model span does not match its text")
                if any(m.start() <= local_start and local_end <= m.end() for m in _CLOCK_RE.finditer(chunk)):
                    continue
                evidence = _model_evidence(category, source, chunk_start + local_start, chunk_start + local_end)
                if evidence == 'reject':
                    continue
                if category == 'date_of_birth' and not _birth_context(source, chunk_start + local_start, chunk_start + local_end):
                    continue
                output.append(_finding(
                    segment_id, source, chunk_start + local_start, chunk_start + local_end,
                    category, score, detector, entity_key, review_only=evidence == 'review' or score < proposal_threshold,
                ))
        return output


def _validate_in_memory_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise _error("Detection policy must be an object")
    # Run the same normalisation checks without requiring a temporary policy file.
    candidate = copy.deepcopy(policy)
    _reject_unknown_policy_keys(candidate)
    if candidate.get("schema_version") != 1:
        raise _error("Detection policy schema_version must be 1")
    merged = copy.deepcopy(_DEFAULT_POLICY)
    merged.update({key: value for key, value in candidate.items() if key not in {"model", "rules", "aliases"}})
    if "model" in candidate:
        if not isinstance(candidate["model"], dict):
            raise _error("Policy model must be an object")
        merged["model"].update(candidate["model"])
    if "rules" in candidate:
        if not isinstance(candidate["rules"], dict):
            raise _error("Policy rules must be an object")
        merged["rules"].update(candidate["rules"])
    merged["aliases"] = _normalise_aliases(candidate.get("aliases", []))
    _normalise_thresholds(merged["model"], candidate.get("model") if isinstance(candidate.get("model"), dict) else None)
    # Reuse file validation by writing no files: the remaining checks are compact.
    labels = merged["model"].get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label for label in labels) or any(label not in _MODEL_CATEGORY for label in labels):
        raise _error("Policy model.labels contains a non-PII or unsupported label")
    _validate_thresholds(merged["model"])
    for key in ("chunk_tokens", "chunk_overlap_tokens"):
        if not isinstance(merged["model"].get(key), int) or merged["model"][key] <= 0:
            raise _error(f"Policy model.{key} must be a positive integer")
    if merged["model"]["chunk_overlap_tokens"] >= merged["model"]["chunk_tokens"]:
        raise _error("Policy model chunk overlap must be smaller than chunk size")
    if any(not isinstance(value, bool) for value in merged["rules"].values()):
        raise _error("Policy rules values must be booleans")
    return merged


def _segment_text(segment: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(segment, dict) or not isinstance(segment.get("id"), str) or not isinstance(segment.get("text"), str):
        raise _error("Each detection segment needs string id and text")
    return segment["id"], segment["text"]


def _integer_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _char_chunks(length: int, size: int, overlap: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    step = max(1, size - overlap)
    start = 0
    while start < length:
        end = min(length, start + size)
        output.append((start, end))
        if end == length:
            break
        start += step
    return output


def rank_passages(segments: list[dict[str, Any]], findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank every retained segment by an uncalibrated missed-PII heuristic."""

    by_segment: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("segment_id"), str):
            raise _error("Invalid finding for passage ranking")
        by_segment.setdefault(finding["segment_id"], []).append(finding)
    ranked: list[dict[str, Any]] = []
    for segment in segments:
        segment_id, text = _segment_text(segment)
        own = by_segment.get(segment_id, [])
        reasons: list[str] = []
        score = 0.0
        if own:
            score += min(0.48, 0.18 * len(own))
            reasons.append("pii_findings_present")
            low = [item for item in own if isinstance(item.get("score"), (int, float)) and item["score"] < 0.75]
            if low:
                score += 0.24
                reasons.append("low_confidence_candidates")
            if len({(item.get("start"), item.get("end")) for item in own}) < len(own):
                score += 0.12
                reasons.append("overlapping_candidates")
            if any(item.get("start", 0) > 0 and item.get("end", len(text)) < len(text) for item in own):
                score += 0.08
                reasons.append("partially_redacted_passage")
        if _CONTEXT_RE.search(text):
            score += 0.22
            reasons.append("personal_context_terms")
        if _EMAIL_RE.search(text) or _IBAN_RE.search(text) or any(_is_phone_candidate(text, match) for match in _PHONE_RE.finditer(text)):
            score += 0.35
            reasons.append("identifier_like_text")
        elif _IDENTIFIER_RE.search(text):
            score += 0.10
            reasons.append("identifier_like_token")
        if not reasons:
            reasons.append("no_candidate_signal")
        ranked.append({"id": segment_id, "score": round(min(1.0, score), 6), "reasons": reasons})
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return ranked
