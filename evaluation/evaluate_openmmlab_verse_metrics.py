#!/usr/bin/env python3
"""Evaluate OpenMMLab semantic/instance checkpoints with the locked VerSe metrics."""

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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


SEMANTIC_CLASS_NAMES = ["background", "cervical", "thoracic", "lumbar"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["semantic", "instance"], required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--verse-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--instance-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def image_files(verse_root: Path, split: str) -> list[Path]:
    image_dir = verse_root / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(image_dir.glob("*.png"))


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def compute_semantic_metrics_from_labels(
    predictions: dict[str, np.ndarray],
    verse_root: Path,
    split: str,
    num_classes: int = 4,
) -> dict[str, Any]:
    gt_dir = verse_root / split / "annotations_semantic"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Semantic GT directory not found: {gt_dir}")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    evaluated_images = 0
    for filename, pred_label in predictions.items():
        gt_path = gt_dir / filename
        if not gt_path.exists():
            continue
        gt = np.asarray(Image.open(gt_path), dtype=np.int64)
        pred_label = np.asarray(pred_label, dtype=np.int64)
        if pred_label.shape != gt.shape:
            pred_label = np.asarray(
                Image.fromarray(pred_label.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), resample=Image.NEAREST
                ),
                dtype=np.int64,
            )
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
    for class_id, class_name in enumerate(SEMANTIC_CLASS_NAMES):
        metrics[f"iou_{class_name}"] = None if np.isnan(iou[class_id]) else float(iou[class_id])
        metrics[f"dice_{class_name}"] = None if np.isnan(dice[class_id]) else float(dice[class_id])
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
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    categories = [
        {"id": int(category["id"]), "name": category.get("name", str(category["id"]))}
        for category in metadata.get("categories", [])
        if int(category.get("id", 0)) != 0
    ]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    return output_json


def compute_coco_segm_metrics(gt_json: Path, pred_json: Path) -> dict[str, Any]:
    coco_gt = COCO(str(gt_json))
    if pred_json.stat().st_size == 0:
        detections: list[dict[str, Any]] = []
    else:
        with pred_json.open("r", encoding="utf-8") as f:
            detections = json.load(f)
    if not detections:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    coco_dt = coco_gt.loadRes(str(pred_json))
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"AP": float(coco_eval.stats[0] * 100), "AP50": float(coco_eval.stats[1] * 100), "AP75": float(coco_eval.stats[2] * 100)}


def compute_instance_match_metrics(
    gt_json: Path,
    pred_json: Path,
    iou_threshold: float,
) -> dict[str, Any]:
    with gt_json.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with pred_json.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    gt_by_image_category: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for annotation in gt_data["annotations"]:
        gt_by_image_category[(annotation["image_id"], annotation["category_id"])].append(annotation)
    pred_by_image_category: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        pred_by_image_category[(pred["image_id"], pred["category_id"])].append(pred)

    gt_instances = len(gt_data["annotations"])
    matched = 0
    dice_scores = []
    for key, gt_list in gt_by_image_category.items():
        pred_list = sorted(pred_by_image_category.get(key, []), key=lambda x: x.get("score", 0.0), reverse=True)
        used_pred = set()
        for gt_ann in gt_list:
            gt_mask = decode_rle(gt_ann["segmentation"])
            best_idx = None
            best_iou = 0.0
            best_dice = 0.0
            for pred_idx, pred in enumerate(pred_list):
                if pred_idx in used_pred:
                    continue
                pred_mask = decode_rle(pred["segmentation"])
                if pred_mask.shape != gt_mask.shape:
                    continue
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = float(intersection / union) if union else 0.0
                if iou > best_iou:
                    denom = gt_mask.sum() + pred_mask.sum()
                    best_dice = float(2 * intersection / denom) if denom else 0.0
                    best_iou = iou
                    best_idx = pred_idx
            if best_idx is not None and best_iou >= iou_threshold:
                used_pred.add(best_idx)
                matched += 1
                dice_scores.append(best_dice)

    return {
        "instance_recall_iou50": float(matched / gt_instances) if gt_instances else 0.0,
        "mean_instance_dice_iou50": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "gt_instances": int(gt_instances),
        "pred_instances": int(len(predictions)),
        "matched_instances_iou50": int(matched),
    }


def write_metrics(metrics: dict[str, Any], output_dir: Path, prefix: str = "verse_metrics") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved metrics CSV: {csv_path}")


def evaluate_semantic(args: argparse.Namespace) -> None:
    from mmseg.apis import inference_model, init_model

    verse_root = Path(args.verse_root)
    output_dir = Path(args.output_dir)
    model = init_model(args.config_file, args.checkpoint, device=args.device)
    predictions: dict[str, np.ndarray] = {}
    for idx, image_path in enumerate(image_files(verse_root, args.split), start=1):
        result = inference_model(model, str(image_path))
        predictions[image_path.name] = result.pred_sem_seg.data.squeeze(0).cpu().numpy()
        if idx % 250 == 0:
            print(f"Semantic inference {idx} images")
    metrics = compute_semantic_metrics_from_labels(predictions, verse_root, args.split)
    metrics.update(
        {
            "task": "semantic",
            "split": args.split,
            "config_file": args.config_file,
            "weights": args.checkpoint,
            "verse_root": str(verse_root),
        }
    )
    write_metrics(metrics, output_dir)


def evaluate_instance(args: argparse.Namespace) -> None:
    from mmdet.apis import inference_detector, init_detector

    verse_root = Path(args.verse_root)
    output_dir = Path(args.output_dir)
    model = init_detector(args.config_file, args.checkpoint, device=args.device)
    # MMDetection filters predictions inside the model before returning pred_instances.
    # Keep this synchronized with the external threshold so a threshold sweep is meaningful.
    if hasattr(model, "test_cfg") and "rcnn" in model.test_cfg:
        model.test_cfg.rcnn.score_thr = args.score_threshold
    if hasattr(model, "cfg") and "model" in model.cfg and "test_cfg" in model.cfg.model:
        if "rcnn" in model.cfg.model.test_cfg:
            model.cfg.model.test_cfg.rcnn.score_thr = args.score_threshold
    with (verse_root / args.split / f"verse_{args.split}_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    image_id_by_name = {image["file_name"]: image["id"] for image in metadata["images"]}

    predictions = []
    for idx, image_path in enumerate(image_files(verse_root, args.split), start=1):
        result = inference_detector(model, str(image_path))
        pred_instances = result.pred_instances
        masks = pred_instances.masks.cpu().numpy() if "masks" in pred_instances else []
        labels = pred_instances.labels.cpu().numpy() if "labels" in pred_instances else []
        scores = pred_instances.scores.cpu().numpy() if "scores" in pred_instances else []
        image_id = image_id_by_name[image_path.name]
        for mask, label, score in zip(masks, labels, scores):
            score = float(score)
            if score < args.score_threshold:
                continue
            binary_mask = np.asarray(mask, dtype=np.uint8)
            if binary_mask.sum() == 0:
                continue
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            bbox = bbox_from_binary_mask(binary_mask)
            if bbox is None:
                continue
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label) + 1,
                    "segmentation": rle,
                    "bbox": bbox,
                    "score": score,
                }
            )
        if idx % 250 == 0:
            print(f"Instance inference {idx} images, predictions={len(predictions)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_json = output_dir / "coco_instances_results.json"
    with pred_json.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)
    gt_json = build_instance_gt_coco(verse_root, args.split, output_dir / "coco_gt_instances.json")
    metrics = {}
    metrics.update(compute_coco_segm_metrics(gt_json, pred_json))
    metrics.update(compute_instance_match_metrics(gt_json, pred_json, args.instance_iou_threshold))
    metrics.update(
        {
            "task": "instance",
            "split": args.split,
            "config_file": args.config_file,
            "weights": args.checkpoint,
            "verse_root": str(verse_root),
        }
    )
    write_metrics(metrics, output_dir)


def main() -> None:
    args = parse_args()
    if args.task == "semantic":
        evaluate_semantic(args)
    else:
        evaluate_instance(args)


if __name__ == "__main__":
    main()
