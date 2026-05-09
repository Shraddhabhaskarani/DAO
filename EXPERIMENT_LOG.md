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
- Fixed `torch.load` in all editable files

## Phase 1: Baseline Establishment (IN PROGRESS)

### 1-shot generation on MP-20 (finetune_mp_20 checkpoint, no energy guidance)
- Started: 2026-05-10
- Generation: 38 batches x 1000 diffusion steps (~6.5 min/batch, ~4 hours total)
- Status: RUNNING
