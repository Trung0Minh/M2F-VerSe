#!/usr/bin/env python3
"""Generate qualitative figures for selected Mask2Former improvement methods."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from generate_qualitative_comparison import (
    colorize_instance,
    colorize_semantic,
    instance_predictions_by_image,
    load_json,
    make_grid,
    normalize_input_image,
    render_instance_predictions,
    select_samples,
    semantic_from_rle_predictions,
)


METHODS = [
    ("Baseline", "baseline"),
    ("2.5D", "p5"),
    ("Focal", "p6a"),
    ("Elastic", "p6b"),
    ("Focal+Elastic", "p6c"),
    ("Multi-window", "p3b"),
]

SEMANTIC_PREDICTIONS = {
    "baseline": Path("Mask2Former/output/experiments/semantic_baseline_multiseed/R50/seed_42/inference/sem_seg_predictions.json"),
    "p5": Path("Mask2Former-p5-2p5d-input/output/experiments/semantic_p5_2p5d_input_multiseed/R50/seed_42/inference/sem_seg_predictions.json"),
    "p6a": Path("Mask2Former-p6a-focal-loss/output/experiments/semantic_p6a_focal_loss_multiseed/R50/seed_42/inference/sem_seg_predictions.json"),
    "p6b": Path("Mask2Former-p6b-elastic-enhancement/output/experiments/semantic_p6b_elastic_multiseed/R50/seed_42/inference/sem_seg_predictions.json"),
    "p6c": Path("Mask2Former-p6c-focal-elastic/output/semantic_p6c_focal_elastic_multiseed/R50/seed_42/inference/sem_seg_predictions.json"),
    "p3b": Path("Mask2Former-p3b-multiwindow/experiments/semantic_p3b_multiwindow_multiseed/R50/seed_42/inference/sem_seg_predictions.json"),
}

INSTANCE_PREDICTIONS = {
    "baseline": Path("Mask2Former/output/experiments/instance_baseline_multiseed/R50/seed_42/inference/coco_instances_results.json"),
    "p5": Path("Mask2Former-p5-2p5d-input/output/experiments/instance_p5_2p5d_input_multiseed/R50/seed_42/inference/coco_instances_results.json"),
    "p6a": Path("Mask2Former-p6a-focal-loss/output/experiments/instance_p6a_focal_loss_multiseed/R50/seed_42/inference/coco_instances_results.json"),
    "p6b": Path("Mask2Former-p6b-elastic-enhancement/output/experiments/instance_p6b_elastic_multiseed/R50/seed_42/inference/coco_instances_results.json"),
    "p6c": Path("Mask2Former-p6c-focal-elastic/output/instance_p6c_focal_elastic_multiseed/R50/seed_42/inference/coco_instances_results.json"),
    "p3b": Path("Mask2Former-p3b-multiwindow/experiments/instance_p3b_multiwindow_multiseed/R50/seed_42/inference/coco_instances_results.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["semantic", "instance", "both"], default="both")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", default="Report TGMT/img")
    parser.add_argument("--min-foreground-area", type=int, default=1200)
    parser.add_argument("--instance-score-threshold", type=float, default=0.0)
    parser.add_argument("--max-instance-preds", type=int, default=35)
    parser.add_argument("--panel-height", type=int, default=512)
    parser.add_argument("--panel-width", type=int, default=384)
    return parser.parse_args()


def ensure_files(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))


def generate_semantic(args: argparse.Namespace) -> Path:
    ensure_files(SEMANTIC_PREDICTIONS)
    verse_root = Path("dataset_verse_2d/ade20k")
    samples = select_samples(verse_root, args.split, args.num_samples, args.min_foreground_area)
    filenames = [sample["image"]["file_name"] for sample in samples]
    image_paths = [verse_root / args.split / "images" / filename for filename in filenames]
    shape_by_name = {
        filename: np.asarray(Image.open(verse_root / args.split / "annotations_semantic" / filename)).shape
        for filename in filenames
    }
    predictions = {
        key: semantic_from_rle_predictions(path, filenames, shape_by_name)
        for key, path in SEMANTIC_PREDICTIONS.items()
    }

    rows = []
    row_labels = []
    for sample, image_path in zip(samples, image_paths):
        filename = sample["image"]["file_name"]
        gt = np.asarray(Image.open(verse_root / args.split / "annotations_semantic" / filename), dtype=np.uint8)
        rows.append(
            [
                normalize_input_image(image_path),
                colorize_semantic(gt),
                *[colorize_semantic(predictions[key][filename]) for _, key in METHODS],
            ]
        )
        row_labels.append(filename.replace(".png", ""))

    output_path = Path(args.output_dir) / "improvement_qualitative_semantic.png"
    make_grid(
        rows,
        row_labels,
        ["Input", "Ground Truth", *[label for label, _ in METHODS]],
        output_path,
        "Semantic Improvement Comparison",
        (args.panel_height, args.panel_width),
    )
    return output_path


def generate_instance(args: argparse.Namespace) -> Path:
    ensure_files(INSTANCE_PREDICTIONS)
    verse_root = Path("dataset_verse_2d/coco")
    samples = select_samples(verse_root, args.split, args.num_samples, args.min_foreground_area)
    metadata = load_json(verse_root / args.split / f"verse_{args.split}_metadata.json")
    annotations_by_image = {ann["image_id"]: ann for ann in metadata["annotations"]}
    predictions = {
        key: instance_predictions_by_image(path)
        for key, path in INSTANCE_PREDICTIONS.items()
    }

    rows = []
    row_labels = []
    for sample in samples:
        image = sample["image"]
        filename = image["file_name"]
        image_id = int(image["id"])
        image_path = verse_root / args.split / "images" / filename
        gt_mask = np.asarray(Image.open(verse_root / args.split / "annotations_instance" / filename), dtype=np.int32)
        shape = gt_mask.shape
        annotation = annotations_by_image[image_id]
        rows.append(
            [
                normalize_input_image(image_path),
                colorize_instance(gt_mask, annotation.get("segments_info", [])),
                *[
                    render_instance_predictions(
                        predictions[key].get(image_id, []),
                        shape,
                        args.instance_score_threshold,
                        args.max_instance_preds,
                    )
                    for _, key in METHODS
                ],
            ]
        )
        row_labels.append(filename.replace(".png", ""))

    output_path = Path(args.output_dir) / "improvement_qualitative_instance.png"
    make_grid(
        rows,
        row_labels,
        ["Input", "Ground Truth", *[label for label, _ in METHODS]],
        output_path,
        "Instance Improvement Comparison",
        (args.panel_height, args.panel_width),
    )
    return output_path


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args()
    if args.task in {"semantic", "both"}:
        generate_semantic(args)
    if args.task in {"instance", "both"}:
        generate_instance(args)


if __name__ == "__main__":
    main()
