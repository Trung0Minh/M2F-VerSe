# Checkpoints

Large pretrained weights and trained checkpoints are not stored in Git.

Use the shared Google Drive folder from the project author for trained model checkpoints:

```text
M2F-VerSe-checkpoints/
```

Shared link:

```text
https://drive.google.com/drive/folders/164nNROVaBk_HRVv6yU_HrNjJQzrw1Vhd?usp=sharing
```

After downloading, place trained checkpoints directly under a local `weights/` folder using this layout:

```text
weights/
├── baseline/
├── focal_loss/
├── elastic_augmentation/
├── focal_elastic/
├── two_point_five_d_input/
├── multi_window/
└── external_comparison/
    ├── semantic/
    └── instance/
```

## Mask2Former checkpoint naming

Use these filenames for Mask2Former checkpoints:

```text
semantic_R50_model_final.pth
semantic_SwinT_model_final.pth
instance_R50_model_final.pth
instance_SwinT_model_final.pth
```

Example:

```text
weights/focal_elastic/semantic_R50_model_final.pth
weights/multi_window/instance_SwinT_model_final.pth
```

## External comparison checkpoints

Use these suggested locations:

```text
weights/external_comparison/semantic/deeplabv3plus_r50.pth
weights/external_comparison/semantic/upernet_r50.pth
weights/external_comparison/instance/mask_rcnn_r50.pth
weights/external_comparison/instance/queryinst_r50.pth
weights/external_comparison/instance/maskdino_r50.pth
```

## Pretrained initialization weights

Pretrained initialization weights are stored separately from trained VerSe checkpoints:

```text
M2F-VerSe-pretrained-weights/
```

Shared link:

```text
https://drive.google.com/open?id=1w0Pg8egEYDYhlAqAEIYkza2sraJYf9Po
```

The folder is organized by model:

```text
M2F-VerSe-pretrained-weights/
├── Mask2Former/
├── DeepLabV3Plus/
├── UPerNet/
├── MaskRCNN/
├── QueryInst/
└── MaskDINO/
```

MaskDINO includes both official checkpoints used for comparison setup:

```text
M2F-VerSe-pretrained-weights/
└── MaskDINO/
    ├── maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth
    └── maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth
```

Mask2Former configs expect ADE20K or COCO initialization weights under each method source folder, for example:

```text
source/Mask2Former-baseline/weights/ade20k_semantic_R50.pkl
source/Mask2Former-baseline/weights/coco_instance_R50.pkl
```

External comparison configs similarly use model-specific pretrained weights from the same folder, such as ADE20K weights for DeepLabV3+/UPerNet/MaskDINO semantic and COCO weights for Mask R-CNN/QueryInst/MaskDINO instance.

These pretrained weights are needed for training from scratch or fine-tuning. They are not needed when evaluating an already trained `model_final.pth`.

## Reproducibility note

The report uses multi-seed results for the selected methods. A single downloaded checkpoint is representative for inference checks, so its exact metric can vary slightly from the reported mean, but it should be consistent with the reported behavior.
