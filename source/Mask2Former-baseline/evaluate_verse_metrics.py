#!/usr/bin/env python3
"""Evaluate a fine-tuned Mask2Former checkpoint on VerSe 2D metrics.

Examples:
    python evaluate_verse_metrics.py \
        --task semantic \
        --config-file configs/verse/verse_ade20k_semantic_R50.yaml \
        --weights output/verse_ade20k_semantic_R50/model_final.pth \
        --verse-root ../dataset_verse_2d/ade20k \
        --split test \
        --output-dir output/eval_semantic_R50

    python evaluate_verse_metrics.py \
        --task instance \
        --config-file configs/verse/verse_coco_instance_R50.yaml \
        --weights output/verse_coco_instance_R50/model_final.pth \
        --verse-root ../dataset_verse_2d/coco \
        --split test \
        --output-dir output/eval_instance_R50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from mask2former.data.datasets.register_dataset import register_all_verse_datasets
from train import Trainer


SEMANTIC_CLASS_NAMES = ["background", "cervical", "thoracic", "lumbar"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["semantic", "instance"], required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True, help="Fine-tuned model_final.pth to evaluate.")
    parser.add_argument("--verse-root", required=True, help="Processed VerSe 2D root for this task.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--instance-iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional Detectron2 config overrides.")
    return parser.parse_args()


def setup_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)

    dataset_name = f"verse_{args.task}_{args.split}"
    opts = [
        "DATASETS.VERSE_ROOT",
        args.verse_root,
        "DATASETS.TEST",
        f'("{dataset_name}",)',
        "MODEL.WEIGHTS",
        args.weights,
        "OUTPUT_DIR",
        args.output_dir,
    ]
    if args.num_classes is not None:
        opts.extend(["MODEL.SEM_SEG_HEAD.NUM_CLASSES", str(args.num_classes)])
    if args.task == "instance":
        opts.extend(
            [
                "MODEL.MASK_FORMER.TEST.SEMANTIC_ON",
                "False",
                "MODEL.MASK_FORMER.TEST.INSTANCE_ON",
                "True",
                "MODEL.MASK_FORMER.TEST.PANOPTIC_ON",
                "False",
            ]
        )
    opts.extend(args.opts)

    cfg.merge_from_list(opts)
    cfg.freeze()

    default_setup(cfg, args)
    register_all_verse_datasets(cfg.DATASETS.VERSE_ROOT)
    return cfg, dataset_name


def run_detectron2_eval(cfg, dataset_name: str) -> dict[str, Any]:
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    return Trainer.test(cfg, model, evaluators=[Trainer.build_evaluator(cfg, dataset_name)])


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def compute_semantic_metrics(
    prediction_json: Path,
    verse_root: Path,
    split: str,
    num_classes: int,
    class_names: list[str],
) -> dict[str, Any]:
    with prediction_json.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    predictions_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        predictions_by_file[Path(pred["file_name"]).name].append(pred)

    gt_dir = verse_root / split / "annotations_semantic"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Semantic GT directory not found: {gt_dir}")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    evaluated_images = 0

    for filename, entries in predictions_by_file.items():
        gt_path = gt_dir / filename
        if not gt_path.exists():
            continue

        gt = np.asarray(Image.open(gt_path), dtype=np.int64)
        pred_label = np.zeros(gt.shape, dtype=np.int64)
        for entry in entries:
            category_id = int(entry["category_id"])
            if category_id < 0 or category_id >= num_classes:
                continue
            mask = decode_rle(entry["segmentation"])
            if mask.shape == pred_label.shape:
                pred_label[mask] = category_id

        valid = (gt != 255) & (gt >= 0) & (gt < num_classes)
        hist = np.bincount(
            num_classes * gt[valid].reshape(-1) + pred_label[valid].reshape(-1),
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)
        confusion += hist
        evaluated_images += 1

    tp = np.diag(confusion).astype(float)
    gt_count = confusion.sum(axis=1).astype(float)
    pred_count = confusion.sum(axis=0).astype(float)
    union = gt_count + pred_count - tp

    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    dice = np.divide(
        2 * tp,
        gt_count + pred_count,
        out=np.full_like(tp, np.nan),
        where=(gt_count + pred_count) > 0,
    )
    acc = np.divide(tp, gt_count, out=np.full_like(tp, np.nan), where=gt_count > 0)

    foreground_ids = list(range(1, num_classes)) if num_classes > 1 else list(range(num_classes))
    metrics: dict[str, Any] = {
        "evaluated_images": int(evaluated_images),
        "pixel_accuracy": float(np.nansum(tp) / max(confusion.sum(), 1)),
        "mean_accuracy": float(np.nanmean(acc)),
        "mIoU": float(np.nanmean(iou)),
        "mean_dice": float(np.nanmean(dice)),
        "foreground_mIoU": float(np.nanmean(iou[foreground_ids])),
        "foreground_mean_dice": float(np.nanmean(dice[foreground_ids])),
    }

    for class_id, class_name in enumerate(class_names):
        metrics[f"iou_{class_name}"] = None if np.isnan(iou[class_id]) else float(iou[class_id])
        metrics[f"dice_{class_name}"] = (
            None if np.isnan(dice[class_id]) else float(dice[class_id])
        )

    return metrics


def bbox_from_binary_mask(binary_mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def build_instance_gt_coco(verse_root: Path, split: str, output_json: Path) -> Path:
    metadata_path = verse_root / split / f"verse_{split}_metadata.json"
    mask_root = verse_root / split / "annotations_instance"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Instance mask directory not found: {mask_root}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    segments_by_image = {
        annotation["image_id"]: annotation.get("segments_info", [])
        for annotation in metadata.get("annotations", [])
    }

    images = []
    annotations = []
    ann_id = 1
    for image_info in metadata.get("images", []):
        file_name = image_info["file_name"]
        mask_path = mask_root / file_name
        if not mask_path.exists():
            continue

        mask_array = np.asarray(Image.open(mask_path))
        images.append(
            {
                "id": image_info["id"],
                "file_name": file_name,
                "width": int(image_info.get("width", mask_array.shape[1])),
                "height": int(image_info.get("height", mask_array.shape[0])),
            }
        )

        for segment in segments_by_image.get(image_info["id"], []):
            category_id = int(segment.get("category_id", 0))
            if category_id == 0:
                continue
            binary_mask = (mask_array == int(segment["id"])).astype(np.uint8)
            area = int(binary_mask.sum())
            if area == 0:
                continue

            bbox = segment.get("bbox") or bbox_from_binary_mask(binary_mask)
            if bbox is None:
                continue

            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_info["id"],
                    "category_id": category_id,
                    "segmentation": rle,
                    "area": area,
                    "bbox": [float(x) for x in bbox],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        category for category in metadata.get("categories", []) if int(category.get("id", 0)) != 0
    ]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    return output_json


def compute_instance_metrics(
    gt_json: Path,
    prediction_json: Path,
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, Any]:
    with prediction_json.open("r", encoding="utf-8") as f:
        predictions = [
            pred for pred in json.load(f) if float(pred.get("score", 1.0)) >= score_threshold
        ]

    ap_metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    coco_gt = COCO(str(gt_json))
    if predictions:
        filtered_prediction_json = prediction_json.with_name("filtered_coco_instances_results.json")
        with filtered_prediction_json.open("w", encoding="utf-8") as f:
            json.dump(predictions, f)
        coco_dt = coco_gt.loadRes(str(filtered_prediction_json))
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_metrics = {
            "AP": float(coco_eval.stats[0] * 100.0),
            "AP50": float(coco_eval.stats[1] * 100.0),
            "AP75": float(coco_eval.stats[2] * 100.0),
        }

    with gt_json.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    pred_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for annotation in gt_data["annotations"]:
        gt_by_key[(annotation["image_id"], annotation["category_id"])].append(annotation)
    for prediction in predictions:
        pred_by_key[(prediction["image_id"], prediction["category_id"])].append(prediction)

    gt_count = sum(len(items) for items in gt_by_key.values())
    matched_count = 0
    dice_scores = []

    for key, gt_items in gt_by_key.items():
        pred_items = pred_by_key.get(key, [])
        if not pred_items:
            continue

        gt_masks = [decode_rle(item["segmentation"]) for item in gt_items]
        pred_masks = [decode_rle(item["segmentation"]) for item in pred_items]
        pairs = []

        for pred_idx, pred_mask in enumerate(pred_masks):
            for gt_idx, gt_mask in enumerate(gt_masks):
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union == 0:
                    continue
                iou = intersection / union
                dice = (2 * intersection) / max(pred_mask.sum() + gt_mask.sum(), 1)
                pairs.append((iou, dice, pred_idx, gt_idx))

        pairs.sort(reverse=True, key=lambda item: item[0])
        used_preds = set()
        used_gts = set()
        for iou, dice, pred_idx, gt_idx in pairs:
            if iou < iou_threshold:
                break
            if pred_idx in used_preds or gt_idx in used_gts:
                continue
            used_preds.add(pred_idx)
            used_gts.add(gt_idx)
            matched_count += 1
            dice_scores.append(float(dice))

    return {
        **ap_metrics,
        "instance_recall_iou50": float(matched_count / gt_count) if gt_count else 0.0,
        "mean_instance_dice_iou50": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "gt_instances": int(gt_count),
        "pred_instances": int(len(predictions)),
        "matched_instances_iou50": int(matched_count),
    }


def write_metrics(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "verse_metrics.json"
    csv_path = output_dir / "verse_metrics.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved metrics CSV: {csv_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cfg, dataset_name = setup_cfg(args)

    with torch.no_grad():
        detectron2_results = run_detectron2_eval(cfg, dataset_name)

    inference_dir = output_dir / "inference"
    metrics: dict[str, Any] = {
        "task": args.task,
        "split": args.split,
        "config_file": str(args.config_file),
        "weights": str(args.weights),
        "verse_root": str(args.verse_root),
    }

    if args.task == "semantic":
        num_classes = args.num_classes or int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        class_names = SEMANTIC_CLASS_NAMES[:num_classes]
        if len(class_names) < num_classes:
            class_names += [f"class_{idx}" for idx in range(len(class_names), num_classes)]
        metrics.update(
            compute_semantic_metrics(
                inference_dir / "sem_seg_predictions.json",
                Path(args.verse_root),
                args.split,
                num_classes,
                class_names,
            )
        )
        sem_seg_results = detectron2_results.get("sem_seg", detectron2_results)
        metrics["detectron2_mIoU"] = sem_seg_results.get("mIoU")
        metrics["detectron2_fwIoU"] = sem_seg_results.get("fwIoU")
        metrics["detectron2_mACC"] = sem_seg_results.get("mACC")
        metrics["detectron2_pACC"] = sem_seg_results.get("pACC")
    else:
        gt_json = build_instance_gt_coco(
            Path(args.verse_root),
            args.split,
            output_dir / f"verse_instance_{args.split}_gt_coco.json",
        )
        metrics.update(
            compute_instance_metrics(
                gt_json,
                inference_dir / "coco_instances_results.json",
                args.instance_iou_threshold,
                args.score_threshold,
            )
        )

    write_metrics(metrics, output_dir)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "42")
    main()
