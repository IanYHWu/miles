# Changelogs

One write-up per major update to this fork: what changed, and the evidence that it works.

- Name each file `YYYY-MM-DD-<short-slug>.md`, dated by when the update was validated. Images go under
  `assets/` with the same date prefix.
- Link every changed file and test relative to this folder (`../miles/...`), so the links work on GitHub
  and in a checkout.
- Each write-up has these sections:
  1. **Summary**: the outcome in two or three sentences.
  2. **What changed**: new or changed flags, then every changed source file with one line on its change.
  3. **Tests**: every new or changed test file, what it covers, and how to run it.
  4. **Validation**: each result with its numbers and what it was compared against.
  5. **Bugs found**: what each broke and the fix.
  6. **Not covered yet**: what the validation leaves out.

| Date | Update | Write-up |
| --- | --- | --- |
| 2026-10-24 | Bounded-memory LM head (`--log-probs-backend fused_lm_head`) and tiled MTP training | [2026-10-24-fused-lm-head-and-mtp.md](2026-10-24-fused-lm-head-and-mtp.md) |
