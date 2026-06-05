# Kaggle Training Notebooks

The notebooks in `notebooks/kaggle/` are for reproducing multi-seed training on Kaggle.

```text
notebooks/
└── kaggle/
    ├── semantic_multiseed_runner.ipynb
    └── instance_multiseed_runner.ipynb
```

They are not required for local inference verification.

## Required Kaggle inputs

Each notebook expects Kaggle datasets for:

- Mask2Former source for the selected method
- VerSe 2D processed data
- Mask2Former configs and initialization weights
- offline Python dependencies

The source path in the notebook config cell should point to the method being tested.

Example method/source mapping:

```text
M2F-VerSe/
├── source/Mask2Former-baseline/                 # baseline
├── source/Mask2Former-focal-loss/               # focal_loss
├── source/Mask2Former-elastic-enhancement/      # elastic_augmentation
├── source/Mask2Former-focal-elastic/            # focal_elastic
├── source/Mask2Former-2p5-input/                # two_point_five_d_input
└── data/multi_window/                           # multi_window data used with baseline source
```

## Multi-seed setup

The report uses repeated runs for selected methods. The seed set used for the final multi-seed tables is:

```text
42, 3407, 2026
```

The notebooks train each requested backbone and seed sequentially, evaluate each final checkpoint, summarize the results, and remove heavy artifacts when configured to do so.

## Task selection

Semantic notebook:

```text
task: semantic
dataset: ade20k-style VerSe 2D
metrics: mIoU, mDice, foreground mIoU, foreground mDice
```

Instance notebook:

```text
task: instance
dataset: COCO-style VerSe 2D
metrics: AP, AP50, AP75, Recall@0.5, Dice@0.5
```

## Save Version note

The notebooks are designed to work with Kaggle Save Version. This lets Kaggle keep running after the browser session is closed and preserves output summaries.

If a run fails, inspect the generated summary JSON/CSV and the per-run `train_command.log` before rerunning.
