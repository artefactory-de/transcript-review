# Development rules

- Read `specs/requirements.md` and `docs/releases.md` before changing behavior or packaging.
- Use synthetic fixtures only. Never add real transcripts, reviewer workbooks,
  mappings, private policies, customer references or machine-specific paths.
- Keep temporary files outside the repository in an explicitly selected directory.
- Run the focused tests, the full pytest suite, Ruff and `git diff --check`.
- Serialize real-model inference and large builds; do not disturb other processes.
- Review exact release files and licences before publication. CI does not replace
  manual functional tests.
