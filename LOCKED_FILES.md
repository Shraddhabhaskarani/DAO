# 🔒 LOCKED FILES - DO NOT EDIT

**Critical Rule for the AutoResearch Loop:**
The autonomous agent **must never edit** any of the files listed below. 
This ensures honest, reproducible, and fair benchmarking against the original DAO SOTA.

### Strictly Locked Files / Directories:

- **All core evaluation code**
  - Any file containing Match Rate or RMSD calculation logic
  - `scripts/` evaluation related scripts (if present)
  - `dao/csp/evaluate.py` (or similar)
  - Any file with "evaluate", "metric", "match_rate", "rmsd" in name/path

- **Test data loaders and benchmark preparation**
  - Files responsible for loading MP-20 and MPTS-52 test sets
  - Any `prepare.py`, `dataset.py` for test/validation splits
  - Data preprocessing pipelines for official benchmarks

- **Original baseline scripts**
  - Any script that reproduces the official DAO paper results
  - Core CLI entry points related to final evaluation

- **Any file that directly affects final reported metrics**
  - Benchmark configuration files that define test sets
  - Structure matching or validity checking code used in final scoring

### Allowed to Edit:
- `experiment_runner.py`
- All files in `conf/` (Hydra configs)
- `dao/models/`
- `dao/training/`
- `dao/generation/`
- Any new files in `src/` or `experiments/`

---

**If you are unsure whether a file is locked:**
- Do **NOT** edit it.
- Check this file first.
- Or only edit files inside the explicitly allowed directories above.

This protection is essential for trustworthy AutoResearch results.

Last Updated: $(date)
