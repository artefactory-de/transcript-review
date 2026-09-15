# Transcript Review

Offline PII detection, Excel review and clean DOCX export for Windows.

1. Select a DOCX and prepare a review.
2. Edit the proposed replacements and ranked passages in the workbook.
3. Import your decisions, sign off the current candidate and export.

Inference runs locally. The default scope is PII only: role names, systems,
processes and generic organisational terms are preserved. Detection can still miss
identifiers or flag ordinary text, so human review is required. Keep review
workspaces private and separate from exports.

See the [user guide](docs/desktop-guide.md), [requirements](specs/requirements.md)
and [release instructions](docs/releases.md).

Pushing a `vX.Y.Z` tag triggers a verified Windows build and publishes the
release automatically if every check passes.

## Development

Use Python 3.12:

```sh
uv sync --locked --extra model --extra build
uv run python scripts/fetch_model.py
uv run transcript-review
uv run python -m pytest
uv run ruff check src scripts tests
```

## Terms

All rights reserved. Use requires separate permission; this is not open-source
software. Provided as is, without support or maintenance commitments.
See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
