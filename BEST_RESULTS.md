# BEST RESULTS - DAO AutoResearch

**Current Best: Original DAO Baseline (finetune_mp_20 checkpoint, no energy guidance)**

### MP-20
- Match Rate (1-shot): 65.95%
- Match Rate (20-shot): TBD
- RMSD: 0.0381
- Avg. E_hull (CHGNet): TBD
- Date: 2026-05-10

### MPTS-52
- Match Rate (1-shot): TBD
- Match Rate (20-shot): TBD
- RMSD: TBD
- Avg. E_hull (CHGNet): TBD
- Date: TBD

**Notes**:
- Baseline uses finetune_mp_20 checkpoint (CrysFormer + DiffCSP, pretrained on CrysDB)
- Paper reports 65.60% for this config (close match, small difference likely from PyTorch version)
- Paper reports 74.17% with CrysFormer + FlowMM (not in this codebase)
- Target: beat 65.95% on 1-shot, then run 20-shot
