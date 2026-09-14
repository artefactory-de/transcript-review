# Releases

Run **Build release candidate** against an existing version tag. The workflow
builds a Windows x64 application with pinned dependencies, model and runtime,
then tests the frozen review/import/sign-off/export cycle.

- **Code ZIP:** executable, documentation, notices and installation metadata.
- **Libraries ZIP:** bundled Python and native libraries.
- **Model assets ZIPs:** checkpoint chunks, each below 900 MiB.

Extract every ZIP from the same release into one new folder. The first model use
verifies and rebuilds the checkpoint locally. Keep at least 3 GiB free for that
one-time step. Builds need at least 8 GiB of free working space.

## Install or update

Extract the Code, Libraries and every Model Assets ZIP into the same fresh
folder. Keep the EXE beside `_internal`, then start it normally. Keep review
folders outside the installation.

## Publish

Build artifacts are retained for three days. Draft upload requires the `draft`
option, `LICENSE`, the `DISTRIBUTION_APPROVED=true` repository variable and a
`release-review` environment with required reviewers. Configure that environment
before enabling draft upload.

The workflow never publishes automatically. Review the exact files, licences,
checksums and functional test results before publishing a draft. Automated tests
do not establish redaction quality or compatibility with every Windows system.
