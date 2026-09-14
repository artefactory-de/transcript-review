# Package requirements

## Input and privacy

- Accept local DOCX without modifying the original. Reject unsupported or unsafe
  document parts; never silently truncate or skip text.
- Rebuild a clean DOCX with speaker text, relative timestamps and source order.
  Remove inherited media, comments, revisions and identifying package metadata.
- Detect PII using pinned local CPU model assets and deterministic identifier rules.
  Preserve generic business terms, roles and systems by default. Birth dates need
  birth context; transcript timestamps and vague scheduling phrases are not birth dates.
- Favor identifier coverage over perfect identity linking. Link clear local variants
  and explicit aliases where evidenced; do not infer arbitrary nickname equivalence.
- No hosted inference, runtime model downloads, telemetry or automatic uploads.
- Validate model labels against identifier syntax and local context. Generic roles,
  field labels and tool names are not identifiers. Weak handle/location evidence
  goes to ranked review, not automatic replacement or alias propagation.

## Review and export

- Generate a source-named, protected XLSX with aggregated proposals and ranked
  retained passages. Explain every action and the whole-passage copy/edit workflow
  within the workbook. Top-align all cells; permit column resizing.
- Only decisions, corrections, notes and sign-off are editable. Independently
  validate imported files, including tampered protection and pasted invalid values.
- Preserve immutable revisions. Rescan corrections and invalidate changed approvals.
  Block stale imports, incompatible detector corrections and unsigned exports.
- Export only the approved clean DOCX and minimal hash manifest. Keep originals,
  mappings and review artifacts local and separate.

## Interface

- Use a small click-to-run desktop with native file/folder dialogs and clear next steps.
- Changing the source clears prior run status, timing, progress and export actions.
  It does not delete old runs. Disabled source controls prevent changes during work.
- Loading, processing and file-creation instructions reflect the current stage.
  Show processing immediately when model loading completes, even before segment one.
- Display elapsed h:mm:ss and an approximate minute-range ETA held for ten-second
  windows. It is not a calibrated confidence interval. Support safe cancellation.

## Distribution and verification

- Full Windows x64 offline ZIP with EXE and sibling support/model folder.
- From the second compatible release, provide an update ZIP for an explicit base
  manifest. Verify every reused/changed file and build a new installation alongside
  the old one. Keep the old installation and review data intact for rollback.
- Pin model hashes, dependency lock, build actions and runtime inputs. Release
  assets include manifests, checksums, upgrade instructions and notices.
- Source tests run on Linux and Windows; native frozen synthetic tests cover review,
  correction, sign-off and export. Only synthetic content enters CI or fixtures.
- Draft release upload and public publication are separate operations. Source
  privacy review, licence/notices review and manual functional tests remain required.
