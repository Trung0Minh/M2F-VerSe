# M2F-VerSe: Mask2Former on VerSe 2D Vertebrae Segmentation

This repository contains the code used to adapt Mask2Former-style segmentation to the VerSe vertebrae dataset.

## What This Repository Is For

- Reproduce local inference/evaluation from trained checkpoints.
- Inspect the Mask2Former source-level improvements.
- Re-run VerSe preprocessing if raw 3D data is available.
- Re-run multi-seed training notebooks on Kaggle if needed.
- Provide custom VerSe configs for external baseline models.

This repository does not include external model source trees such as MMDetection, MMSegmentation, or MaskDINO. Those are used from their official repositories.

## Detailed Guides

Recommended reading order for local verification:

1. [Environment setup](envs/README.md) # Create Conda environments and run required post-install steps.
2. [Checkpoints and weights](docs/CHECKPOINTS.md) # Download trained checkpoints and place them under `weights/`.
3. [Data preparation](docs/DATA_PREPARATION.md) # Understand the expected processed VerSe 2D dataset layout.
4. [Run inference](docs/RUN_INFERENCE.md) # Reproduce metrics from downloaded checkpoints.
5. [Repository structure](docs/REPO_STRUCTURE.md) # Understand what each folder contains and what source code is intentionally excluded.
6. [Kaggle training notebooks](docs/KAGGLE_TRAINING_NOTEBOOKS.md) # Optional: reproduce multi-seed training with Kaggle Save Version.

## Acknowledgements and Citations

This project is a refactoring and adaptation built mainly from two upstream works.

### VerSe Dataset

The 3D CT data and vertebrae annotations come from the VerSe benchmark.

- Official repository: <https://github.com/anjany/verse>
- Paper: Sekuboyina A. et al., "VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images", Medical Image Analysis, 2021.

### Mask2Former

The baseline architecture, training framework, and Detectron2-style implementation are adapted from Mask2Former.

- Official repository: <https://github.com/facebookresearch/Mask2Former>
- Paper: Cheng B. et al., "Masked-attention Mask Transformer for Universal Image Segmentation", CVPR, 2022.

## Repository Tree

```text
M2F-VerSe/
├── README.md                                      # Main reviewer-facing guide and entry point.
│
├── envs/                                         # Conda environment definitions.
│   ├── README.md                                 # Which environment to use for each model family.
│   ├── verse_detectron2.yml                      # Detectron2/Mask2Former environment.
│   └── verse_openmmlab.yml                       # OpenMMLab environment for comparison models.
│
├── source/                                       # Self-contained Mask2Former source variants.
│   ├── Mask2Former-baseline/                     # Baseline adapted to VerSe semantic/instance tasks.
│   ├── Mask2Former-focal-loss/                   # Focal-loss variant.
│   ├── Mask2Former-elastic-enhancement/          # Elastic/geometric augmentation variant.
│   ├── Mask2Former-focal-elastic/                # Combined focal loss + elastic augmentation variant.
│   └── Mask2Former-2p5-input/                    # Adjacent-slice 2.5D input variant.
│
├── configs_for_external_comparison/              # VerSe configs for external model repos.
│   ├── semantic/                                 # DeepLabV3+, UPerNet, MaskDINO semantic configs.
│   └── instance/                                 # Mask R-CNN, QueryInst, MaskDINO instance configs.
│
├── evaluation/                                   # Metric and qualitative evaluation scripts.
│   ├── evaluate_openmmlab_verse_metrics.py       # Metrics for OpenMMLab semantic/instance outputs.
│   ├── evaluate_prediction_json_verse_metrics.py # Metrics from saved prediction JSON files.
│   ├── generate_qualitative_comparison.py        # Qualitative comparison across baseline models.
│   └── generate_improvement_qualitative_comparison.py # Qualitative comparison for improvements.
│
├── notebooks/                                    # Preprocessing and Kaggle training notebooks.
│   ├── preprocessing/                            # VerSe 3D-to-2D conversion notebooks.
│   │   ├── process_verse_2d.ipynb                # Baseline 2D VerSe export.
│   │   ├── process_verse_2d_multiwindow.ipynb    # Multi-window data-level export.
│   │   ├── data_utilities.py                     # Shared preprocessing helpers.
│   │   └── sample/                               # Small sample subject for quick checks.
│   └── kaggle/                                   # Kaggle Save-Version multi-seed notebooks.
│       ├── semantic_multiseed_runner.ipynb       # Semantic multi-seed training/evaluation.
│       └── instance_multiseed_runner.ipynb       # Instance multi-seed training/evaluation.
│
├── docs/                                         # Detailed setup and reproduction guides.
│   ├── REPO_STRUCTURE.md                         # Source policy and folder organization.
│   ├── DATA_PREPARATION.md                       # Raw VerSe and processed 2D dataset layouts.
│   ├── CHECKPOINTS.md                            # Drive links and checkpoint/weight placement.
│   ├── RUN_INFERENCE.md                          # Local evaluation commands.
│   └── KAGGLE_TRAINING_NOTEBOOKS.md              # Kaggle multi-seed workflow.
│
├── data/                         # Not committed. Raw VerSe data, if available.
├── dataset_verse_2d/             # Not committed. Baseline processed 2D data.
├── weights/                      # Not committed. Downloaded checkpoints/weights.
└── output/                       # Not committed. Generated evaluation outputs.
```

## Setup Overview

### 1. Create Conda Environments

Use `verse_detectron2` for all Detectron2 ecosystem methods:

```bash
conda env create -f envs/verse_detectron2.yml
conda activate verse_detectron2
```

Use `verse_openmmlab` for external OpenMMLab comparison models:

```bash
conda env create -f envs/verse_openmmlab.yml
conda activate verse_openmmlab
```

See `envs/README.md` for source installation notes and CUDA compatibility details.

### 2. Download Data and Checkpoints

Download links and local placement rules are documented in:

- `docs/DATA_PREPARATION.md` for raw and processed VerSe data;
- `docs/CHECKPOINTS.md` for trained checkpoints and pretrained initialization weights.

Expected local folders:

```text
M2F-VerSe/
├── dataset_verse_2d/
│   ├── ade20k/                    # Semantic segmentation data.
│   └── coco/                      # Instance segmentation data.
│
└── weights/
    ├── baseline/                  # Trained Mask2Former baseline checkpoints.
    ├── focal_loss/                # Trained Focal Loss checkpoints.
    ├── elastic_augmentation/      # Trained elastic augmentation checkpoints.
    ├── focal_elastic/             # Trained Focal Loss + elastic checkpoints.
    ├── 2p5d_input/                # Trained 2.5D input checkpoints.
    ├── multiwindow/               # Trained multi-window checkpoints.
    └── comparison_models/         # Trained external-model checkpoints.
```

### 3. Compile Mask2Former CUDA Operators

Before running Mask2Former inference or training, first follow the Detectron2 post-install step in `envs/README.md`. Then compile the custom multi-scale deformable attention operators inside the source folder being used:

```bash
cd source/Mask2Former-baseline/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

Repeat for other source variants if running them directly.

### 4. Run Local Inference

After the environment, data, checkpoints, and CUDA operators are ready, run the commands in `docs/RUN_INFERENCE.md`.

## Local Inference Example

For Mask2Former-based methods, run from the selected source folder:

```bash
conda activate verse_detectron2
cd source/Mask2Former-baseline
python -u evaluate_verse_metrics.py \
  --task semantic \
  --config-file configs/verse/verse_ade20k_semantic_R50.yaml \
  --weights ../../weights/baseline/semantic_R50_model_final.pth \
  --verse-root ../../dataset_verse_2d/ade20k \
  --split test \
  --output-dir output/eval_semantic_R50
```

For all evaluation commands, including instance segmentation and external models, see `docs/RUN_INFERENCE.md`.

## Improvement Methods

The report studies the following selected methods:

```text
M2F-VerSe/
├── source/Mask2Former-baseline/                 # Baseline.
├── source/Mask2Former-focal-loss/               # Focal Loss.
├── source/Mask2Former-elastic-enhancement/      # Elastic augmentation.
├── source/Mask2Former-focal-elastic/            # Focal Loss + elastic augmentation.
├── source/Mask2Former-2p5-input/                # 2.5D adjacent-slice input.
└── data/multi_window/                           # Multi-window input data used with baseline source.
```

The multi-window method changes the data representation, not the Mask2Former source code.

## External Comparison Models

External model source code is intentionally excluded for repository size and licensing clarity. Use the official repositories, then apply the configs in `configs_for_external_comparison/`.

```text
External comparison models:
├── semantic/                       # DeepLabV3+, UPerNet, MaskDINO semantic.
└── instance/                       # Mask R-CNN, QueryInst, MaskDINO instance.
```

MaskDINO is also treated as an external repository. Use its official source tree together with the VerSe configs in `configs_for_external_comparison/`.

## Kaggle Training Notebooks

The notebooks in `notebooks/kaggle/` are designed for Kaggle Save Version and multi-seed experiments. They are not required for a teacher/reviewer who only wants to verify trained-checkpoint metrics locally.

See `docs/KAGGLE_TRAINING_NOTEBOOKS.md` for the exact workflow.

## Artifact Policy

Do not commit generated artifacts:

- `data/`, `dataset_verse_2d/`, `weights/`, `output/`;
- `*.pth`, `*.pkl`;
- TensorBoard event files;
- temp inference/training folders;
- compiled caches and build folders.

These are ignored by `.gitignore` and should be downloaded or regenerated when needed.
