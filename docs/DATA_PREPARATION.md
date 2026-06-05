# Data Preparation

The project uses 2D slices derived from the VerSe dataset.

## Baseline processed data

Baseline processed data is expected at:

```text
dataset_verse_2d/
├── ade20k/
│   ├── train/
│   ├── val/
│   └── test/
└── coco/
    ├── train/
    ├── val/
    └── test/
```

`ade20k/` is used for semantic segmentation. `coco/` is used for instance segmentation.

The processed baseline 2D dataset can be downloaded directly from:

```text
https://drive.google.com/drive/folders/11SJXxFGo206_QetjwQ6Jq0ipWXydevpK?usp=sharing
```

Place the downloaded folder as:

```text
M2F-VerSe/dataset_verse_2d/
```

## Multi-window processed data

The multi-window method stores its processed data at:

```text
data/
└── multi_window/
    ├── ade20k/
    └── coco/
```

This data-level method uses the baseline Mask2Former source and changes only the input data representation.

The processed multi-window 2D dataset can be downloaded directly from:

```text
https://drive.google.com/drive/folders/1GQmchc43gj_R2v9uxOB_Jn7ri7SglpuA?usp=sharing
```

Place the downloaded folder as:

```text
M2F-VerSe/data/multi_window/
```

## Preprocessing notebooks

Use:

```text
notebooks/
└── preprocessing/
    ├── process_verse_2d.ipynb
    └── process_verse_2d_multiwindow.ipynb
```

The preprocessing folder also contains:

```text
notebooks/
└── preprocessing/
    ├── data_utilities.py
    └── sample/
```

The sample case is included so the preprocessing pipeline can be checked quickly without processing the full dataset.

## Expected labels

Semantic segmentation uses four classes:

```text
background, cervical, thoracic, lumbar
```

Instance segmentation uses vertebra labels from C1 through L6 plus Sacrum, Cocc, and T13 where present in the dataset annotation.
