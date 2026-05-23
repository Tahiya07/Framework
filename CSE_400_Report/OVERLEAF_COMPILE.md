# Overleaf compile timeout — fixes applied

## What we fixed in the project

1. **Compressed large diagrams** (~6 MB PNGs → ~1.2 MB JPEGs): `hightlevelarch`, `dfd`, `flowchart`, `er`, `blockdiagram`.
2. **Removed duplicate `\label{fig:system_architecture}`** (six identical labels confused hyperref and slowed builds).
3. **Removed unused Bloom/cross-domain images** not referenced in the report.
4. **`main.tex`**: duplicate `longtable` package removed; `hyperref` uses `hidelinks`.

## If Overleaf still times out

### Option A — Fast compile (no List of Figures/Tables)

In `main.tex` line 4, change to:

```latex
\fastcompiletrue
```

Recompile once. Then set back to `\fastcompilefalse` for the final PDF.

### Option B — Overleaf settings

- **Menu → Settings → Compiler timeout** → increase (Premium: up to 10 min).
- **Recompile from scratch**: delete auxiliary files (Logs and output files → Clear cached files).
- Use **pdfLaTeX** (not LaTeX → dvipdf).

### Option C — Upload fresh zip

Use `CSE_400_Report_Merged.zip` from Downloads (already contains compressed images).

## Expected compile time

After compression, a full compile is typically **1–3 minutes** on Overleaf free tier instead of timing out on 5+ MB raw PNGs.
