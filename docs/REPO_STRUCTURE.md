# Repository Structure

This repository separates project code by purpose so a reviewer can run evaluation without searching through old experiment outputs.

## Main folders

```text
source/
  Mask2Former-baseline/              Baseline Mask2Former source
  Mask2Former-focal-loss/            Mask2Former with focal loss
  Mask2Former-elastic-enhancement/   Mask2Former with elastic augmentation
  Mask2Former-focal-elastic/         Mask2Former with focal loss + elastic augmentation
  Mask2Former-2p5-input/             Mask2Former with 2.5D adjacent-slice input

configs_for_external_comparison/
  semantic/                          VerSe configs for DeepLabV3+ and UPerNet
  instance/                          VerSe configs for Mask R-CNN, QueryInst, and MaskDINO

notebooks/
  preprocessing/                     VerSe 2D preprocessing notebooks and sample case
  kaggle/                            Multi-seed Kaggle training notebooks

envs/                                Conda environment files for local setup
evaluation/                          Evaluation and qualitative visualization scripts
docs/                                Local setup, checkpoint, and notebook guides
```

## Source policy

Each Mask2Former method in `source/` is self-contained and keeps its own `train.py`, `evaluate_verse_metrics.py`, `configs/`, and `mask2former/` package.

The multi-window method is data-level, so it does not have a separate Mask2Former source. Run it with `source/Mask2Former-baseline/` and point `DATASETS.VERSE_ROOT` to `data/multi_window/ade20k` or `data/multi_window/coco`.

External comparison model sources are not included in this repository. Use the official upstream repositories for DeepLabV3+, UPerNet, Mask R-CNN, QueryInst, and MaskDINO, then apply the custom VerSe configs from `configs_for_external_comparison/`.

Processed datasets are generated locally and are not committed:

```text
dataset_verse_2d/                    Baseline processed VerSe 2D data
data/multi_window/                   Multi-window processed VerSe 2D data
```

## Generated files

Do not commit large generated artifacts:

- `output/`
- `experiments/`
- `work_dirs/`
- `queue_logs/`
- `*.pth`
- `*.pkl`
- TensorBoard event files

Download trained checkpoints from the shared Drive folder described in `docs/CHECKPOINTS.md`.
