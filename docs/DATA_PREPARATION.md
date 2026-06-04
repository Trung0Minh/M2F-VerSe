# Data Preparation

The project uses 2D slices derived from the VerSe dataset.

## Baseline processed data

Baseline processed data is expected at:

```text
dataset_verse_2d/
  ade20k/
    train/
    val/
    test/
  coco/
    train/
    val/
    test/
```

`ade20k/` is used for semantic segmentation. `coco/` is used for instance segmentation.

## Multi-window processed data

The multi-window method stores its processed data at:

```text
data/multi_window/
  ade20k/
  coco/
```

This data-level method uses the baseline Mask2Former source and changes only the input data representation.

## Preprocessing notebooks

Use:

```text
notebooks/preprocessing/process_verse_2d.ipynb
notebooks/preprocessing/process_verse_2d_multiwindow.ipynb
```

The preprocessing folder also contains:

```text
notebooks/preprocessing/data_utilities.py
notebooks/preprocessing/sample/
```

The sample case is included so the preprocessing pipeline can be checked quickly without processing the full dataset.

## Expected labels

Semantic segmentation uses four classes:

```text
background, cervical, thoracic, lumbar
```

Instance segmentation uses vertebra labels from C1 through L6 plus Sacrum, Cocc, and T13 where present in the dataset annotation.
