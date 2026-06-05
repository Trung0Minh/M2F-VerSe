# M2F-VerSe: Mask2Former on VerSe 2D Vertebrae Segmentation

This repository contains the code used to adapt Mask2Former-style segmentation to the VerSe vertebrae dataset. It includes the Mask2Former variants used in the report, preprocessing notebooks, evaluation scripts, Kaggle multi-seed training notebooks, environment files, and VerSe configs for external comparison models.

Large datasets, pretrained initialization weights, and trained checkpoints are not committed to Git. They are provided through the Drive links documented in `docs/CHECKPOINTS.md` and `docs/DATA_PREPARATION.md`.

## What This Repository Is For

- Reproduce local inference/evaluation from trained checkpoints.
- Inspect the Mask2Former source-level improvements.
- Re-run VerSe preprocessing if raw 3D data is available.
- Re-run multi-seed training notebooks on Kaggle if needed.
- Provide custom VerSe configs for external baseline models.

This repository does not include external model source trees such as MMDetection, MMSegmentation, or MaskDINO. Those are used from their official repositories.

## Detailed Guides

- [Repository structure](docs/REPO_STRUCTURE.md) # What each folder contains and what source code is intentionally excluded.
- [Data preparation](docs/DATA_PREPARATION.md) # How raw VerSe data is converted into the 2D semantic, instance, and multi-window datasets.
- [Checkpoints and weights](docs/CHECKPOINTS.md) # Drive links and expected local placement for pretrained weights and trained checkpoints.
- [Run inference](docs/RUN_INFERENCE.md) # Commands for reproducing metrics from downloaded checkpoints.
- [Kaggle training notebooks](docs/KAGGLE_TRAINING_NOTEBOOKS.md) # How the multi-seed Kaggle Save-Version notebooks were used.
- [Environment setup](envs/README.md) # Which Conda environment to use for Mask2Former and OpenMMLab models.

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
├── requirements.txt                              # Shared package pins for the Mask2Former-side setup.
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
dataset_verse_2d/
  ade20k/                         # Semantic segmentation data
  coco/                           # Instance segmentation data

weights/
  checkpoints/                    # Trained model_final checkpoints
```

### 3. Compile Mask2Former CUDA Operators

Before running Mask2Former inference or training, compile the custom multi-scale deformable attention operators inside the source folder being used:

```bash
cd source/Mask2Former-baseline/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

Repeat for other source variants if running them directly.

## Local Inference Example

For Mask2Former-based methods, run from the selected source folder:

```bash
cd source/Mask2Former-baseline
conda run --no-capture-output -n verse_detectron2 python -u evaluate_verse_metrics.py \
  --task semantic \
  --config-file configs/verse/verse_ade20k_semantic_R50.yaml \
  --weights ../../weights/checkpoints/baseline/semantic_R50_model_final.pth \
  --verse-root ../../dataset_verse_2d/ade20k \
  --split test \
  --output-dir output/eval_semantic_R50
```

For all evaluation commands, including instance segmentation and external models, see `docs/RUN_INFERENCE.md`.

## Improvement Methods

The report studies the following selected methods:

```text
Baseline                  source/Mask2Former-baseline/
Focal loss                source/Mask2Former-focal-loss/
Elastic augmentation      source/Mask2Former-elastic-enhancement/
Focal + elastic           source/Mask2Former-focal-elastic/
2.5D adjacent input       source/Mask2Former-2p5-input/
Multi-window input        baseline source + multi-window processed data
```

The multi-window method changes the data representation, not the Mask2Former source code.

## External Comparison Models

External model source code is intentionally excluded for repository size and licensing clarity. Use the official repositories, then apply the configs in `configs_for_external_comparison/`.

```text
Semantic comparison:
  DeepLabV3+
  UPerNet
  MaskDINO semantic

Instance comparison:
  Mask R-CNN
  QueryInst
  MaskDINO instance
```

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
