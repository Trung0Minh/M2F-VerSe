# Kaggle Multi-Seed Training Notebooks

These notebooks are Kaggle proof/reproduction utilities. They are not the main local verification path for the report.

Use:

- `semantic_multiseed_runner.ipynb` for semantic segmentation.
- `instance_multiseed_runner.ipynb` for instance segmentation.

On Kaggle, attach the public datasets listed below, edit only the first configuration cell if needed, then run all cells. The notebooks train each selected backbone and seed, evaluate the final checkpoint, aggregate mean/std metrics, and remove heavy model artifacts from `/kaggle/working`.

## Required Kaggle Inputs

Attach these shared datasets for all methods:

| Purpose | Kaggle dataset path used in notebook |
|---|---|
| Configs and pretrained weights | `/kaggle/input/datasets/trungminh815/m2f-configs-and-weights` |
| Offline Python dependencies | `/kaggle/input/datasets/trungminh815/m2f-offline-dependencies` |
| Baseline semantic data | `/kaggle/input/datasets/trungminh815/verse2d/ade20k` |
| Baseline instance data | `/kaggle/input/datasets/trungminh815/verse2d/coco` |
| Multi-window semantic data | `/kaggle/input/datasets/trungminh815/verse2d-p3b/ade20k` |
| Multi-window instance data | `/kaggle/input/datasets/trungminh815/verse2d-p3b/coco` |

## Method Config Guide

Set `SOURCE_INPUT_DIR`, `EXPERIMENT_NAME`, `SOURCE_TAG`, `VERSE_ROOT`, and `COMMON_EXTRA_OPTS` in the first notebook cell.

| Method | Source input | Semantic `VERSE_ROOT` | Instance `VERSE_ROOT` | `COMMON_EXTRA_OPTS` |
|---|---|---|---|---|
| Baseline | `/kaggle/input/datasets/trungminh815/source-m2f` | `/kaggle/input/datasets/trungminh815/verse2d/ade20k` | `/kaggle/input/datasets/trungminh815/verse2d/coco` | `[]` |
| 2.5D input | `/kaggle/input/datasets/trungminh815/source-m2f-p5-2p5d-input` | baseline semantic data | baseline instance data | `['INPUT.USE_2P5D', 'True', 'INPUT.NEIGHBOR_SLICE_DELTA', '1']` |
| Focal loss | `/kaggle/input/datasets/trungminh815/source-m2f-p6a-focal-loss` | baseline semantic data | baseline instance data | `['MODEL.MASK_FORMER.CLASS_LOSS_TYPE', 'focal', 'MODEL.MASK_FORMER.FOCAL_ALPHA', '0.25', 'MODEL.MASK_FORMER.FOCAL_GAMMA', '2.0']` |
| Elastic augmentation | `/kaggle/input/datasets/trungminh815/source-m2f-p6b-elastic-enhancement` | baseline semantic data | baseline instance data | `['INPUT.ELASTIC.ENABLED', 'True', 'INPUT.ELASTIC.PROB', '0.5', 'INPUT.ELASTIC.ROTATION_DEGREES', '10.0', 'INPUT.ELASTIC.SCALE_MIN', '0.9', 'INPUT.ELASTIC.SCALE_MAX', '1.1', 'INPUT.ELASTIC.TRANSLATE_FRAC', '0.05', 'INPUT.ELASTIC.ALPHA', '20.0', 'INPUT.ELASTIC.SIGMA', '5.0']` |
| Focal + elastic | `/kaggle/input/datasets/trungminh815/source-m2f-p6c-focal-elastic` | baseline semantic data | baseline instance data | focal options plus elastic options above |
| Multi-window data | `/kaggle/input/datasets/trungminh815/source-m2f` | `/kaggle/input/datasets/trungminh815/verse2d-p3b/ade20k` | `/kaggle/input/datasets/trungminh815/verse2d-p3b/coco` | `[]` |

For multi-window data, the source remains baseline Mask2Former. The processed VerSe dataset changes.

## Notes

- The report uses seeds `[42, 3407, 2026]` for the locked multi-seed results.
- To run fewer backbones, edit `BACKBONES`, for example `['R50']` or `['R50', 'SwinT']`.
- Kaggle Save Version can preserve summaries/logs after the session ends. These notebooks are designed to continue writing summary files even when one backbone fails.
