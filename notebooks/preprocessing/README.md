# VerSe Preprocessing Notebooks

These notebooks convert raw VerSe 3D CT volumes and vertebra masks into 2D datasets used by the segmentation experiments.

## Notebooks

| Notebook | Output | Purpose |
|---|---|---|
| `process_verse_2d.ipynb` | `dataset_verse_2d/` | Baseline 2D VerSe export for semantic, instance, and panoptic-style formats. |
| `process_verse_2d_multiwindow.ipynb` | `dataset_verse_2d_multiwindow/` | Data-level multi-window variant where RGB channels encode different CT windows. |

## Expected Raw Data Layout

Both notebooks expect raw VerSe data under the repository root:

```text
data/
├── train/
│   ├── rawdata/sub-*/
│   └── derivatives/sub-*/
├── val/
│   ├── rawdata/sub-*/
│   └── derivatives/sub-*/
└── test/
    ├── rawdata/sub-*/
    └── derivatives/sub-*/
```

The notebooks resolve the repository root automatically, so they can be launched from the repo root or from `notebooks/preprocessing/`.

## Notes

- `process_verse_2d.ipynb` is the original baseline processing pipeline.
- `process_verse_2d_multiwindow.ipynb` keeps the same slice filtering and labels but changes image channels to multi-window CT input.
- Generated datasets are large and should not be committed to Git.
