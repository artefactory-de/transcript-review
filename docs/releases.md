# Releases

Push a version tag such as `v1.2.3`. The workflow builds a Windows x64
application with pinned dependencies, model and runtime, tests the frozen
review/import/sign-off/export cycle, then publishes the release if every check
passes. On a later release it automatically uses the most recent public Full ZIP
as the update base. Run **Build release candidate** manually only when an
unpublished candidate is needed; manual candidate runs do not publish.

- **Code ZIP:** executable, documentation, notices and installation metadata.
- **Libraries ZIP:** bundled Python and native libraries.
- **Model assets ZIPs:** checkpoint chunks, each below 900 MiB.
- **Full ZIP:** the complete application in one archive, for approved delivery
  channels that permit files larger than 1 GiB.

Extract every ZIP from the same release into one new folder. The first model use
verifies and rebuilds the checkpoint locally. Keep at least 3 GiB free for that
one-time step. Builds need at least 8 GiB of free working space.

## Install or update

Either extract the Full ZIP into a fresh folder, or extract the Code, Libraries
and every Model Assets ZIP into the same fresh folder. Keep the EXE beside
`_internal`, then start it normally. Keep review folders outside the
installation.

## Publish

Build artifacts are retained for three days. Every pushed version tag uploads the
verified files as a public release. Automated tests do not establish redaction
quality or compatibility with every Windows system.
