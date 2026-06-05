# Run Inference and Evaluation

This guide is for local verification of trained checkpoints. It does not require retraining.

Download trained checkpoints from the trained-checkpoint Drive folder in `docs/CHECKPOINTS.md`. The separate pretrained-weights Drive folder is only needed for training or fine-tuning.

## Environments

Use the `verse_detectron2` Conda environment for Mask2Former-based methods.

Use the `verse_openmmlab` Conda environment for OpenMMLab comparison models such as DeepLabV3+, UPerNet, Mask R-CNN, and QueryInst.

Activate the matching environment before running the commands. This is preferred over wrapping commands with `conda run`, because CUDA/NVML availability can be inconsistent on some systems when long GPU scripts are launched through `conda run`.

## Mask2Former evaluation

Run commands from the method source folder.

Semantic R50 example:

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

Instance R50 example:

```bash
conda activate verse_detectron2
cd source/Mask2Former-baseline
python -u evaluate_verse_metrics.py \
  --task instance \
  --config-file configs/verse/verse_coco_instance_R50.yaml \
  --weights ../../weights/baseline/instance_R50_model_final.pth \
  --verse-root ../../dataset_verse_2d/coco \
  --split test \
  --output-dir output/eval_instance_R50
```

For Swin-T, use:

```text
configs/verse/verse_ade20k_semantic_swin_t.yaml
configs/verse/verse_coco_instance_swin_t.yaml
semantic_SwinT_model_final.pth
instance_SwinT_model_final.pth
```

## Selected improvements

Use the same command pattern with a different source folder and checkpoint folder:

```text
M2F-VerSe/
├── source/Mask2Former-focal-loss/              # weights/focal_loss/
├── source/Mask2Former-elastic-enhancement/     # weights/elastic_augmentation/
├── source/Mask2Former-focal-elastic/           # weights/focal_elastic/
└── source/Mask2Former-2p5-input/               # weights/2p5d_input/
```

For multi-window, use the baseline source with multi-window data:

```bash
conda activate verse_detectron2
cd source/Mask2Former-baseline
python -u evaluate_verse_metrics.py \
  --task semantic \
  --config-file configs/verse/verse_ade20k_semantic_R50.yaml \
  --weights ../../weights/multiwindow/semantic_R50_model_final.pth \
  --verse-root ../../data/multi_window/ade20k \
  --split test \
  --output-dir output/eval_multi_window_semantic_R50
```

Use `../../data/multi_window/coco` for instance segmentation.

## External comparison models

External comparison model source code is not committed in this repository. Use the official upstream repositories for the relevant framework/model, then run with the custom VerSe configs stored in:

```text
configs_for_external_comparison/
```

Some external configs contain relative `_base_` paths. If the external framework cannot resolve a moved config directly, copy the config into the corresponding external repo `configs/verse/` folder before running, or adjust its base paths to match your local clone.

DeepLabV3+ example:

```bash
conda activate verse_openmmlab
python -u evaluation/evaluate_openmmlab_verse_metrics.py \
  --task semantic \
  --config-file configs_for_external_comparison/semantic/deeplabv3plus_r50_verse.py \
  --checkpoint weights/comparison_models/semantic/deeplabv3plus_R50_iter_20000.pth \
  --verse-root dataset_verse_2d/ade20k \
  --split test \
  --output-dir output/eval_semantic_deeplabv3plus_r50 \
  --device cuda:0
```

Mask R-CNN example:

```bash
conda activate verse_openmmlab
python -u evaluation/evaluate_openmmlab_verse_metrics.py \
  --task instance \
  --config-file configs_for_external_comparison/instance/mask-rcnn_r50_verse.py \
  --checkpoint weights/comparison_models/instance/mask_rcnn_R50_iter_20000.pth \
  --verse-root dataset_verse_2d/coco \
  --split test \
  --output-dir output/eval_instance_mask_rcnn_r50 \
  --device cuda:0
```

## Outputs

Evaluation writes metric files under the chosen `--output-dir`, usually including:

```text
verse_metrics.json
verse_metrics.csv
```

These files can be regenerated from the checkpoint and dataset.
