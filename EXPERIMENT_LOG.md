# EXPERIMENT LOG - DAO AutoResearch Loop

All iterations and results will be recorded here.

---

**AutoResearch Loop Started**: 2026-05-10

## Phase 0: Environment Bootstrap (COMPLETE)

- Created conda env `dao` with Python 3.10.16
- PyTorch 2.12.0.dev20260408+cu128 (nightly, Blackwell sm_120 support)
- PyG 2.7.0, Lightning 2.6.1, pymatgen 2025.10.7
- Downloaded checkpoints: dao_g, dao_p, finetune_mp_20, finetune_mpts_52
- Downloaded MP-20 dataset from CDVAE repo (27136 train, 9047 val, 9046 test)
- MPTS-52 not yet available (needs matbench_genmetrics)

## Compatibility Fixes (COMPLETE)

- Replaced `torch_scatter` CUDA ops with PyG native `scatter`/`softmax` (scatter_compat.py)
- Fixed Lightning 2.x: `gpus`->`accelerator`/`devices`, removed `progress_bar_refresh_rate`, `resume_from_checkpoint`, `terminate_on_nan`
- Fixed PyG imports: `DataLoader` from `torch_geometric.loader`
- Patched `torch.load` to default `weights_only=False` via `_bootstrap.py`
- Created `diffcsp`->`dao` module compatibility shim for legacy checkpoints
- Fixed batch_size reduction for energy guidance (batch_size=100 to avoid OOM)
- Fixed idx_pool logic to cover all test structures when batch_size changes

## Phase 1: Baseline Establishment

### 1-shot Baseline (finetune_mp_20 checkpoint, no energy guidance)
- Date: 2026-05-10
- Config: batch_size=240, 38 batches, 1 eval, 1000 PC-sampling steps
- **Results: Match Rate 65.95%, RMSD 0.0381**
- Paper reports 65.60% for this config (close match)

### Energy-Guided 1-shot (standard constant aug=20.0)
- Date: 2026-05-10
- Attempted with batch_size=40: completed but only covered 1520/9046 structures (idx_pool bug)
- Killed full run with batch_size=100 after 4h40m (too slow, ~8+ hours estimated)
- **Status: SKIPPED** — moved directly to adaptive guidance (Iteration 1)

## Phase 2: Iterative Improvements

### Iteration 1: Adaptive Energy Guidance (IN PROGRESS)
- **Change**: Replace constant `aug=20.0` with timestep-dependent `aug_t = aug * (1 + 2.0 * (1 - t_norm))`
  - Rationale: `std_x²` and `sigmas²` decay to near-zero at late timesteps, making guidance weakest when it matters most (final denoising). Adaptive scaling compensates by ramping from 1x (start) to 3x (end).
  - Also adds gradient clipping: `torch.clamp(grad, -1.0, 1.0)`
- **Files**: `dao/pl_modules/FTModels.py:360-379`, `dao/pl_modules/PTModels.py:316-335`
- **Status**: 5-batch test PASSED. Full generation launched (91 batches @ batch_size=100, ~5.7 hours)
- **ETA**: ~6:30 AM 2026-05-11

### Iteration 2: Enable Energy Loss During Finetuning (PENDING)
- **Change**: Set `finetune_energy: True` in `conf/model/finetune.yaml`
- **Status**: Waiting for Iteration 1 generation to complete (GPU occupied)

### Iteration 3: Coordinate-Focused Loss Weighting (PENDING)
- **Change**: `cost_coord: 2.0`, `cost_lattice: 0.5` in `conf/model/finetune.yaml`
- **Rationale**: Match Rate depends primarily on coordinate accuracy

### Iteration 4: Differential Learning Rate (CODED, NOT TESTED)
- **Change**: Override `configure_optimizers()` in `CrystFinetuneModel`
  - Early layers (0-5): lr/10
  - Mid layers (6-8): lr/5
  - Late layers (9-11): lr/2
  - Output heads: full lr
- **Files**: `dao/pl_modules/FTModels.py` (new `configure_optimizers` method)

### Iteration 5: Final LayerNorm (READY TO APPLY)
- **Change**: `ln: true` in `conf/model/decoder/crysformer.yaml:14`
- **Rationale**: Modern transformers universally use final LayerNorm for training stability

### Iteration 6: Space Group Conditioning (CODED, NOT TESTED)
- **Change**: Add `nn.Embedding(231, hidden_dim)` to CrysFormer, wire through data pipeline
- **Files**: `dao/pl_modules/crysformer.py` (sg_embedding), `dao/pl_data/dataset.py` (CSV spacegroup lookup), `dao/pl_modules/FTModels.py` (pass spacegroup to decoder)
- **Note**: Preprocessed data reduces all structures to P1. Script reads original spacegroup from CSV instead.
- **To activate**: Set `use_space_group: true` in `conf/data/mp_20.yaml`

### Iteration 7: CHGNet Post-Relaxation (CODED, NOT TESTED)
- **Script**: `scripts/run/relax_chgnet.py`
- **Usage**: `python scripts/run/relax_chgnet.py --input eval_diff_1_adaptive.pt`
- **Depends on**: `chgnet` package (installed v0.4.2)

## BEST RESULTS

| Config | Match Rate (1-shot) | RMSD | Date |
|--------|--------------------:|------:|------|
| Baseline (no EG) | 65.95% | 0.0381 | 2026-05-10 |
| Adaptive EG (Iter 1) | TBD | TBD | Running |
