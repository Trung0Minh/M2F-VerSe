import os
import sys
import torch
import numpy as np
import json
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to allow importing mask2former
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, Metadata
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

# IMPORTANT: Import the registration script
import mask2former.data.datasets.register_dataset

def get_parser():
    parser = argparse.ArgumentParser(description="Visualize Mask2Former Panoptic Segmentation on VerSe")
    parser.add_argument("--config", default="configs/verse/verse_ade20k_panoptic_R50.yaml", help="path to config file")
    parser.add_argument("--weights", default="output/verse_ade20k_panoptic_R50/model_final.pth", help="path to trained weights")
    parser.add_argument("--input", help="path to a single input image. If empty, picks a random test image.")
    parser.add_argument("--output", default="results/verse_ade20k_panoptic_R50_result.png", help="output image path")
    parser.add_argument("--metadata", default="../dataset_verse_2d/ade20k/test/verse_test_metadata.json", help="path to GT metadata JSON")
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
    cfg.freeze()

    # 1.1 Register Dataset
    if cfg.DATASETS.VERSE_ROOT:
        mask2former.data.datasets.register_dataset.register_all_verse_datasets(cfg.DATASETS.VERSE_ROOT)

    print(f"Loading panoptic model with weights: {cfg.MODEL.WEIGHTS}")
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else "verse_panoptic_test")

    # 2. Pick a sample
    if args.input:
        img_path = args.input
        sample_name = os.path.basename(img_path)
    else:
        test_img_dir = "../dataset_verse_2d/ade20k/test/images"
        sample_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
        sample_name = random.choice(sample_images)
        img_path = os.path.join(test_img_dir, sample_name)

    gt_pan_path = img_path.replace("images", "ade20k_panoptic")
    
    # Load Ground Truth Metadata
    gt_segments = []
    if os.path.exists(args.metadata):
        with open(args.metadata, 'r') as f:
            gt_data = json.load(f)
        if "annotations" in gt_data:
            try:
                gt_annos = next(a for a in gt_data["annotations"] if a.get("file_name", "") == sample_name)
                gt_segments = gt_annos.get("segments_info", [])
            except StopIteration:
                pass

    print(f"Visualizing panoptic slice: {sample_name}")

    # 3. Predict
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1] 
    outputs = predictor(img_bgr)
    panoptic_seg, pred_segments = outputs["panoptic_seg"]

    # 4. Create "Silent" Metadata to stop overlapping labels
    silent_metadata = Metadata()
    for k, v in metadata.as_dict().items():
        if k not in ["name", "thing_classes", "stuff_classes"]:
            setattr(silent_metadata, k, v)
    silent_metadata.thing_classes = ["" for _ in range(len(metadata.thing_classes))]
    silent_metadata.stuff_classes = ["" for _ in range(len(metadata.stuff_classes))]

    # 5. Visualize Prediction using silent metadata
    v = Visualizer(img_rgb, metadata=silent_metadata, scale=1.0)
    out_pred = v.draw_panoptic_seg(panoptic_seg.to("cpu"), pred_segments)

    # 6. Load Ground Truth Panoptic Mask
    has_gt = os.path.exists(gt_pan_path)
    if has_gt:
        gt_pan_img = np.array(Image.open(gt_pan_path))
        gt_ids = gt_pan_img[:, :, 0].astype(np.int32) + gt_pan_img[:, :, 1].astype(np.int32) * 256

    # 7. Plotting
    num_panels = 3 if has_gt else 2
    fig, ax = plt.subplots(1, num_panels, figsize=(8 * num_panels, 10))

    ax[0].imshow(img_rgb)
    ax[0].set_title(f"Original Image: {sample_name}", fontsize=15)

    if has_gt:
        ax[1].imshow(img_rgb)
        ax[1].imshow(np.where(gt_ids > 0, gt_ids, np.nan), cmap='tab20', alpha=0.6)
        ax[1].set_title("Ground Truth (Blue Labels)", fontsize=15)
        
        # MANUAL LABELS ON GROUND TRUTH
        for seg in gt_segments:
            mask = (gt_ids == seg["id"])
            y, x = np.where(mask)
            if len(x) > 0:
                cx, cy = np.mean(x), np.mean(y)
                c_idx = seg["category_id"] - 1 if seg["category_id"] > 0 else 0
                label = metadata.thing_classes[c_idx] if seg["isthing"] else "bg"
                ax[1].text(cx, cy, label, color='white', fontsize=9, fontweight='bold',
                          bbox=dict(facecolor='blue', alpha=0.6, pad=0.2, edgecolor='none'),
                          ha='center', va='center')
        pred_ax = ax[2]
    else:
        pred_ax = ax[1]

    pred_ax.imshow(out_pred.get_image())
    pred_ax.set_title("Model Prediction (Black Labels)", fontsize=15)

    # MANUAL LABELS ON PREDICTION
    for seg in pred_segments:
        mask = (panoptic_seg == seg["id"]).cpu().numpy()
        y, x = np.where(mask)
        if len(x) > 0:
            cx, cy = np.mean(x), np.mean(y)
            c_idx = seg["category_id"] - 1 if seg["isthing"] else 0
            label = metadata.thing_classes[c_idx] if seg["isthing"] else "bg"
            pred_ax.text(cx, cy, label, color='white', fontsize=9, fontweight='bold',
                      bbox=dict(facecolor='black', alpha=0.6, pad=0.2, edgecolor='none'),
                      ha='center', va='center')

    for a in ax: a.axis('off')
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Success! Result saved as '{args.output}'")
    plt.show()

