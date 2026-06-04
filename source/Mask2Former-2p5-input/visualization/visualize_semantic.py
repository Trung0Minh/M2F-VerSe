import os
import sys
import torch
import numpy as np
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to allow importing mask2former
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import mask2former.data.datasets.register_dataset

def get_parser():
    parser = argparse.ArgumentParser(description="Visualize Mask2Former Semantic Segmentation on VerSe")
    parser.add_argument("--config", default="configs/verse/verse_ade20k_semantic_R50.yaml", help="path to config file")
    parser.add_argument("--weights", default="output/verse_ade20k_semantic_R50/model_final.pth", help="path to trained weights")
    parser.add_argument("--input", help="path to a single input image. If empty, picks a random test image.")
    parser.add_argument("--output", default="results/verse_ade20k_semantic_R50_result.png", help="output image path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to run model on")
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

    print(f"Loading semantic model with weights: {cfg.MODEL.WEIGHTS}")
    predictor = DefaultPredictor(cfg)

    # 2. Pick a sample
    if args.input:
        img_path = args.input
        sample_name = os.path.basename(img_path)
    else:
        test_img_dir = "../dataset_verse_2d/ade20k/test/images"
        sample_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
        sample_name = random.choice(sample_images)
        img_path = os.path.join(test_img_dir, sample_name)

    gt_path = img_path.replace("images", "annotations_semantic")
    print(f"Visualizing slice: {sample_name}")

    # 3. Predict
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1] 
    outputs = predictor(img_bgr)
    prediction = outputs["sem_seg"].argmax(0).cpu().numpy()

    # 4. Load Ground Truth
    has_gt = os.path.exists(gt_path)
    if has_gt:
        gt = np.array(Image.open(gt_path))

    # 5. Plotting
    num_panels = 3 if has_gt else 2
    fig, ax = plt.subplots(1, num_panels, figsize=(8 * num_panels, 10))

    ax[0].imshow(img_rgb)
    ax[0].set_title(f"Input: {sample_name}", fontsize=15)

    custom_cmap = ListedColormap(['black', 'red', 'green', 'blue'])
    # For overlay, we want background (0) to be transparent
    overlay_cmap = ListedColormap(['none', 'red', 'green', 'blue'])
    
    labels = ['Background', 'Cervical', 'Thoracic', 'Lumbar']
    colors = ['black', 'red', 'green', 'blue']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]

    if has_gt:
        ax[1].imshow(img_rgb)
        ax[1].imshow(gt, cmap=overlay_cmap, vmin=0, vmax=3, alpha=0.5)
        ax[1].set_title("Ground Truth (Manual)", fontsize=15)
        pred_ax = ax[2]
    else:
        pred_ax = ax[1]

    pred_ax.imshow(img_rgb)
    pred_ax.imshow(prediction, cmap=overlay_cmap, vmin=0, vmax=3, alpha=0.5)
    pred_ax.set_title("Model Prediction", fontsize=15)
    pred_ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    for a in ax: a.axis('off')
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Success! Result saved as '{args.output}'")
    plt.show()
