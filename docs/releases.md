# Releases

Run **Build release candidate** against an existing version tag. The workflow
builds a Windows x64 application with pinned dependencies, model and runtime,
then tests the frozen review/import/sign-off/export cycle.

- **Full ZIP:** complete offline application, manifests, checksums and notices.
- **Update ZIP:** changed files for the specified `base_tag`. CI verifies that
  applying it reconstructs the full release. The first release is full-only.

Updates may be large when dependencies or model assets change. Each ZIP must be
under 2 GiB. Builds need at least 8 GiB of free working space.

## Install or update

Extract a full ZIP and keep the EXE beside `_internal`.
For an update, close the application, extract the update ZIP separately and run
`Apply-Update.exe`. Select the previous installation. A verified new installation
is created beside it; the old one remains available for rollback. Keep review
folders outside both installations.

## Publish

Build artifacts are retained for three days. Draft upload requires the `draft`
option, `LICENSE`, the `DISTRIBUTION_APPROVED=true` repository variable and a
`release-review` environment with required reviewers. Configure that environment
before enabling draft upload.

The workflow never publishes automatically. Review the exact files, licences,
checksums and functional test results before publishing a draft. Automated tests
do not establish redaction quality or compatibility with every Windows system.
