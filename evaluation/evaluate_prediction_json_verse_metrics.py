#!/usr/bin/env python3
"""Compute locked VerSe metrics from saved prediction JSON files.

This is useful for Detectron2-style models such as MaskDINO after eval-only has
already produced sem_seg_predictions.json or coco_instances_results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

from evaluate_openmmlab_verse_metrics import (
    SEMANTIC_CLASS_NAMES,
    build_instance_gt_coco,
    compute_coco_segm_metrics,
    compute_instance_match_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["semantic", "instance"], required=True)
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--verse-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--instance-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def compute_semantic_metrics_from_rle(
    prediction_json: Path,
    verse_root: Path,
    split: str,
    num_classes: int,
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
    for gt_path in sorted(gt_dir.glob("*.png")):
        gt = np.asarray(Image.open(gt_path), dtype=np.int64)
        pred_label = np.zeros(gt.shape, dtype=np.int64)
        for entry in predictions_by_file.get(gt_path.name, []):
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
    foreground_ids = list(range(1, num_classes))

    metrics: dict[str, Any] = {
        "evaluated_images": int(evaluated_images),
        "pixel_accuracy": float(np.nansum(tp) / max(confusion.sum(), 1)),
        "mean_accuracy": float(np.nanmean(acc)),
        "mIoU": float(np.nanmean(iou)),
        "mean_dice": float(np.nanmean(dice)),
        "foreground_mIoU": float(np.nanmean(iou[foreground_ids])),
        "foreground_mean_dice": float(np.nanmean(dice[foreground_ids])),
    }
    for class_id, class_name in enumerate(SEMANTIC_CLASS_NAMES[:num_classes]):
        metrics[f"iou_{class_name}"] = None if np.isnan(iou[class_id]) else float(iou[class_id])
        metrics[f"dice_{class_name}"] = None if np.isnan(dice[class_id]) else float(dice[class_id])
    return metrics


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
    prediction_json = Path(args.prediction_json)
    verse_root = Path(args.verse_root)
    output_dir = Path(args.output_dir)

    if args.task == "semantic":
        metrics = compute_semantic_metrics_from_rle(
            prediction_json, verse_root, args.split, args.num_classes
        )
    else:
        gt_json = build_instance_gt_coco(
            verse_root, args.split, output_dir / "coco_gt_instances.json"
        )
        metrics = {}
        metrics.update(compute_coco_segm_metrics(gt_json, prediction_json))
        metrics.update(
            compute_instance_match_metrics(
                gt_json, prediction_json, args.instance_iou_threshold
            )
        )

    metrics.update(
        {
            "task": args.task,
            "split": args.split,
            "prediction_json": str(prediction_json),
            "verse_root": str(verse_root),
        }
    )
    write_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
