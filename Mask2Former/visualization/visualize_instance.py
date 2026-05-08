import os
import torch
import numpy as np
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

# IMPORTANT: Import the registration script to fill the MetadataCatalog
import mask2former.data.datasets.register_dataset

def get_parser():
    parser = argparse.ArgumentParser(description="Visualize Mask2Former Instance Segmentation on VerSe")
    parser.add_argument("--config", default="configs/verse/verse_instance_R50.yaml", help="path to config file")
    parser.add_argument("--weights", default="output/verse_instance_R50/model_final.pth", help="path to trained weights")
    parser.add_argument("--input", help="path to a single input image. If empty, picks a random test image.")
    parser.add_argument("--output", default="results/verse_instance_R50_result.png", help="output image path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to run model on")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="confidence threshold")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    # 1. Setup Configuration
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.conf_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.conf_threshold
    cfg.freeze()

    # 1.1 Register Dataset
    if cfg.DATASETS.VERSE_ROOT:
        mask2former.data.datasets.register_dataset.register_all_verse_datasets(cfg.DATASETS.VERSE_ROOT)

    print(f"Loading instance model with weights: {cfg.MODEL.WEIGHTS}")
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else "verse_instance_test")

    # 2. Pick a sample
    if args.input:
        img_path = args.input
        sample_name = os.path.basename(img_path)
    else:
        test_img_dir = "../dataset_verse_2d/ade20k/test/images"
        sample_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
        sample_name = random.choice(sample_images)
        img_path = os.path.join(test_img_dir, sample_name)

    gt_path = img_path.replace("images", "annotations_instance")
    print(f"Visualizing slice: {sample_name}")

    # 3. Predict
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1] 
    outputs = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")

    # Filter for confidence
    keep = instances.scores > args.conf_threshold
    instances = instances[keep]

    # --- VISUALIZATION LOGIC ---
    v = Visualizer(img_rgb, metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)

    if not instances.has("pred_boxes") or len(instances.pred_boxes) == 0:
        from detectron2.structures import Boxes
        masks = instances.pred_masks.numpy()
        boxes = []
        for mask in masks:
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                boxes.append([np.min(x), np.min(y), np.max(x), np.max(y)])
            else:
                boxes.append([0, 0, 0, 0])
        instances.pred_boxes = Boxes(torch.tensor(boxes))

    out_pred = v.draw_instance_predictions(instances)
    pred_img = out_pred.get_image()

    # 5. Load Ground Truth
    has_gt = os.path.exists(gt_path)
    if has_gt:
        gt_mask = np.array(Image.open(gt_path))
    
    # 6. Plotting
    num_panels = 3 if has_gt else 2
    fig, ax = plt.subplots(1, num_panels, figsize=(8 * num_panels, 10))

    ax[0].imshow(img_rgb)
    ax[0].set_title(f"Original Image: {sample_name}", fontsize=15)

    if has_gt:
        ax[1].imshow(img_rgb)
        ax[1].imshow(np.where(gt_mask > 0, gt_mask, np.nan), cmap='tab20', alpha=0.6)
        ax[1].set_title("Ground Truth (Manual)", fontsize=15)
        pred_ax = ax[2]
    else:
        pred_ax = ax[1]

    pred_ax.imshow(pred_img)
    pred_ax.set_title("Model Predictions", fontsize=15)

    # MANUAL LABELS
    for i in range(len(instances)):
        mask = instances.pred_masks[i].numpy()
        y, x = np.where(mask)
        if len(x) > 0:
            cx, cy = np.mean(x), np.mean(y)
            label = metadata.thing_classes[instances.pred_classes[i]]
            pred_ax.text(cx, cy, label, color='white', fontsize=11, fontweight='bold',
                      bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'),
                      ha='center', va='center')

    for a in ax: a.axis('off')
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Success! Result saved as '{args.output}'")
    plt.show()

