# CSE 400 Report — merged submission package

This folder merges your prior report (`CSE_400_G_19_4_1_.zip`) with updated evaluation results from the Framework repository.

## Included from your prior report

- Student names, supervisor, co-supervisor, November 2025 title page
- Stakeholder requirements, architecture diagrams, CEP/CEA tables
- Cross-domain Bloom figures, privacy pareto, detailed QA/OCR tables
- Acknowledgements and examiner certificate text

## Updated in this merge

- **LoRA Bloom results** (official test: 0.748 accuracy) + confusion matrix
- **Hold-out baselines** (SVM 0.839, zero-shot 0.441) from `evaluation_outputs/`
- Methodology: LoRA training path documented
- Abstract, conclusion, appendices aligned with measured metrics
- Removed: Chapter Guidelines chapter, sample IoT budget, template-only files

## Build PDF (Overleaf or local)

```bash
cd CSE_400_Report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Refresh metrics from code

```bash
python build_bloom_comparison.py
```
