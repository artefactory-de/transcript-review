# User guide

## Prepare

Extract the full ZIP and double-click the EXE. Keep its support folder beside it.
No Python installation or model download is needed.

Select a DOCX and click **Prepare review**. The default workspace is a
`Transcript review` folder beside the input. It contains sensitive text and
mappings: keep it private. The original DOCX is not changed.

Progress shows the current stage, elapsed h:mm:ss and an approximate minute-range
ETA. The estimate updates in ten-second windows; it is not a guarantee.
Cancellation takes effect at a safe processing boundary.

## Review

Open the source-named workbook and follow **Start here**. Review the proposed
replacements, then the ranked retained passages. Edit only the marked fields;
column widths are adjustable. To correct multiple items in a passage, copy its
text into the correction field and edit each item there. Word edits are not imported.

Save and close Excel, then import the workbook. Content changes produce an updated
workbook and clear sign-off. Review that workbook, fill in **Signed off by**, save,
close and import again.

After a detection update, prepare a new review from the original DOCX. Existing
reviews are not automatically rescanned.

## Export

Export is available only for the signed-off current candidate. It creates
`transcript.docx` and `manifest.json` in a separate folder. Do not include the
original, review workbook or mappings when sharing the export.

Detection can miss identifiers or redact ordinary text. Review remains necessary.
If Windows blocks an unsigned executable, do not disable security controls.
