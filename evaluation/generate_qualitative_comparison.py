#!/usr/bin/env python3
"""Generate qualitative VerSe prediction comparison figures.

The script creates report-ready side-by-side figures for the R50 models used in
Section 4. It reuses saved prediction JSON files when available and only runs
OpenMMLab semantic inference for the selected slices.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


SEMANTIC_COLORS = np.array(
    [
        [0, 0, 0],
        [0, 114, 189],
        [217, 83, 25],
        [119, 172, 48],
    ],
    dtype=np.uint8,
)

INSTANCE_COLORS = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["semantic", "instance", "both"], default="both")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", default="Report TGMT/img")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-foreground-area", type=int, default=1200)
    parser.add_argument("--instance-score-threshold", type=float, default=0.0)
    parser.add_argument("--max-instance-preds", type=int, default=35)
    parser.add_argument("--panel-height", type=int, default=512)
    parser.add_argument("--panel-width", type=int, default=384)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def colorize_semantic(label_map: np.ndarray) -> np.ndarray:
    label_map = np.asarray(label_map, dtype=np.int64)
    safe = np.clip(label_map, 0, len(SEMANTIC_COLORS) - 1)
    return SEMANTIC_COLORS[safe]


def color_for_category(category_id: int) -> np.ndarray:
    return INSTANCE_COLORS[(int(category_id) - 1) % len(INSTANCE_COLORS)]


def colorize_instance(mask: np.ndarray, segments: list[dict[str, Any]] | None = None) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    if segments:
        for segment in segments:
            segment_id = int(segment["id"])
            category_id = int(segment.get("category_id", segment_id))
            output[mask == segment_id] = color_for_category(category_id)
    else:
        for instance_id in sorted(int(x) for x in np.unique(mask) if int(x) != 0):
            output[mask == instance_id] = color_for_category(instance_id)
    return output


def normalize_input_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return np.stack([image, image, image], axis=-1)


def foreground_area_from_metadata(annotation: dict[str, Any]) -> int:
    return int(sum(seg.get("area", 0) for seg in annotation.get("segments_info", [])))


def select_samples(verse_root: Path, split: str, num_samples: int, min_area: int) -> list[dict[str, Any]]:
    metadata = load_json(verse_root / split / f"verse_{split}_metadata.json")
    annotations_by_image = {ann["image_id"]: ann for ann in metadata["annotations"]}
    candidates = []
    for image in metadata["images"]:
        annotation = annotations_by_image.get(image["id"])
        if not annotation:
            continue
        area = foreground_area_from_metadata(annotation)
        segment_count = len(annotation.get("segments_info", []))
        height = int(image.get("height", 0))
        width = int(image.get("width", 0))
        image_path = verse_root / split / "images" / image["file_name"]
        if area >= min_area and segment_count >= 14 and height <= 1000 and width >= 250 and image_path.exists():
            candidates.append(
                {
                    "image": image,
                    "annotation": annotation,
                    "area": area,
                    "segment_count": segment_count,
                }
            )
    if len(candidates) < num_samples:
        raise RuntimeError(
            f"Only found {len(candidates)} samples with foreground area >= {min_area}; "
            "lower --min-foreground-area."
        )
    by_subject: dict[str, dict[str, Any]] = {}
    for item in sorted(candidates, key=lambda item: item["area"], reverse=True):
        subject = item["image"]["file_name"].split("_sag_")[0]
        by_subject.setdefault(subject, item)
    diverse = list(by_subject.values())
    diverse = sorted(diverse, key=lambda item: item["area"], reverse=True)
    return diverse[:num_samples]


def fit_to_panel(image: np.ndarray, panel_size: tuple[int, int]) -> np.ndarray:
    panel_h, panel_w = panel_size
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    h, w = image.shape[:2]
    scale = min(panel_w / max(w, 1), panel_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resample = Image.NEAREST
    resized = Image.fromarray(image.astype(np.uint8)).resize((new_w, new_h), resample=resample)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    y0 = (panel_h - new_h) // 2
    x0 = (panel_w - new_w) // 2
    panel[y0 : y0 + new_h, x0 : x0 + new_w] = np.asarray(resized, dtype=np.uint8)
    return panel


def semantic_from_rle_predictions(prediction_json: Path, filenames: list[str], shape_by_name: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    predictions = load_json(prediction_json)
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        by_file[Path(pred["file_name"]).name].append(pred)

    outputs = {}
    for filename in filenames:
        label = np.zeros(shape_by_name[filename], dtype=np.uint8)
        for pred in by_file.get(filename, []):
            category_id = int(pred["category_id"])
            mask = decode_rle(pred["segmentation"])
            if mask.shape == label.shape:
                label[mask] = category_id
        outputs[filename] = label
    return outputs


def generate_openmmlab_semantic_predictions(
    model_specs: dict[str, tuple[Path, Path]],
    image_paths: list[Path],
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    from mmseg.apis import inference_model, init_model

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for model_name, (config_file, checkpoint) in model_specs.items():
        model = init_model(str(config_file), str(checkpoint), device=device)
        model_outputs = {}
        for image_path in image_paths:
            result = inference_model(model, str(image_path))
            model_outputs[image_path.name] = result.pred_sem_seg.data.squeeze(0).cpu().numpy()
        outputs[model_name] = model_outputs
    return outputs


def instance_predictions_by_image(prediction_json: Path) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in load_json(prediction_json):
        by_image[int(pred["image_id"])].append(pred)
    return by_image


def render_instance_predictions(
    predictions: list[dict[str, Any]],
    shape: tuple[int, int],
    score_threshold: float,
    max_preds: int,
) -> np.ndarray:
    output = np.zeros((*shape, 3), dtype=np.uint8)
    filtered = [p for p in predictions if float(p.get("score", 0.0)) >= score_threshold]
    filtered = sorted(filtered, key=lambda p: float(p.get("score", 0.0)), reverse=True)[:max_preds]
    for pred in reversed(filtered):
        mask = decode_rle(pred["segmentation"])
        if mask.shape != shape:
            continue
        color = color_for_category(int(pred["category_id"]))
        output[mask] = color
    return output


def make_grid(
    rows: list[list[np.ndarray]],
    row_labels: list[str],
    col_labels: list[str],
    output_path: Path,
    title: str,
    panel_size: tuple[int, int],
) -> None:
    nrows = len(rows)
    ncols = len(col_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.45 * nrows), squeeze=False)
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            ax = axes[r][c]
            ax.imshow(fit_to_panel(image, panel_size))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=11, fontweight="bold", rotation=90, labelpad=12)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(pad=0.1, w_pad=0.02, h_pad=0.08, rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def generate_semantic(args: argparse.Namespace) -> Path:
    verse_root = Path("dataset_verse_2d/ade20k")
    samples = select_samples(verse_root, args.split, args.num_samples, args.min_foreground_area)
    filenames = [sample["image"]["file_name"] for sample in samples]
    image_paths = [verse_root / args.split / "images" / filename for filename in filenames]
    shape_by_name = {
        filename: np.asarray(Image.open(verse_root / args.split / "annotations_semantic" / filename)).shape
        for filename in filenames
    }

    openmmlab_specs = {
        "DeepLabV3+": (
            Path("semantic/mmsegmentation/configs/verse/deeplabv3plus_r50_verse.py"),
            Path("output/verse_semantic_deeplabv3plus/iter_20000.pth"),
        ),
        "UPerNet": (
            Path("semantic/mmsegmentation/configs/verse/upernet_r50_verse.py"),
            Path("output/verse_semantic_upernet/iter_20000.pth"),
        ),
    }
    missing = [str(path) for spec in openmmlab_specs.values() for path in spec if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing semantic model files:\n" + "\n".join(missing))

    openmmlab_predictions = generate_openmmlab_semantic_predictions(openmmlab_specs, image_paths, args.device)
    mask2former_predictions = semantic_from_rle_predictions(
        Path("Mask2Former/output/verse_ade20k_semantic_R50/inference/sem_seg_predictions.json"),
        filenames,
        shape_by_name,
    )
    maskdino_predictions = semantic_from_rle_predictions(
        Path("instance/MaskDINO/output/eval_semantic_R50_test/inference/sem_seg_predictions.json"),
        filenames,
        shape_by_name,
    )

    rows = []
    row_labels = []
    for sample, image_path in zip(samples, image_paths):
        filename = sample["image"]["file_name"]
        gt = np.asarray(Image.open(verse_root / args.split / "annotations_semantic" / filename), dtype=np.uint8)
        rows.append(
            [
                normalize_input_image(image_path),
                colorize_semantic(gt),
                colorize_semantic(maskdino_predictions[filename]),
                colorize_semantic(openmmlab_predictions["DeepLabV3+"][filename]),
                colorize_semantic(openmmlab_predictions["UPerNet"][filename]),
                colorize_semantic(mask2former_predictions[filename]),
            ]
        )
        row_labels.append(filename.replace(".png", ""))

    output_path = Path(args.output_dir) / "qualitative_semantic.png"
    make_grid(
        rows,
        row_labels,
        ["Input", "Ground Truth", "MaskDINO", "DeepLabV3+", "UPerNet", "Mask2Former"],
        output_path,
        "Semantic Segmentation",
        (args.panel_height, args.panel_width),
    )
    return output_path


def generate_instance(args: argparse.Namespace) -> Path:
    verse_root = Path("dataset_verse_2d/coco")
    samples = select_samples(verse_root, args.split, args.num_samples, args.min_foreground_area)
    metadata = load_json(verse_root / args.split / f"verse_{args.split}_metadata.json")
    annotations_by_image = {ann["image_id"]: ann for ann in metadata["annotations"]}

    model_predictions = {
        "Mask R-CNN": instance_predictions_by_image(Path("output/eval_instance_mask_rcnn_r50/coco_instances_results.json")),
        "MaskDINO": instance_predictions_by_image(Path("instance/MaskDINO/output/eval_instance_R50_test/inference/coco_instances_results.json")),
        "QueryInst": instance_predictions_by_image(Path("output/eval_instance_queryinst_r50/coco_instances_results.json")),
        "Mask2Former": instance_predictions_by_image(Path("Mask2Former/temp/eval_instance_R50/inference/coco_instances_results.json")),
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
                render_instance_predictions(model_predictions["Mask R-CNN"].get(image_id, []), shape, args.instance_score_threshold, args.max_instance_preds),
                render_instance_predictions(model_predictions["MaskDINO"].get(image_id, []), shape, args.instance_score_threshold, args.max_instance_preds),
                render_instance_predictions(model_predictions["QueryInst"].get(image_id, []), shape, args.instance_score_threshold, args.max_instance_preds),
                render_instance_predictions(model_predictions["Mask2Former"].get(image_id, []), shape, args.instance_score_threshold, args.max_instance_preds),
            ]
        )
        row_labels.append(filename.replace(".png", ""))

    output_path = Path(args.output_dir) / "qualitative_instance.png"
    make_grid(
        rows,
        row_labels,
        ["Input", "Ground Truth", "Mask R-CNN", "MaskDINO", "QueryInst", "Mask2Former"],
        output_path,
        "Instance Segmentation",
        (args.panel_height, args.panel_width),
    )
    return output_path


def main() -> None:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args()
    if args.task in {"semantic", "both"}:
        generate_semantic(args)
    if args.task in {"instance", "both"}:
        generate_instance(args)


if __name__ == "__main__":
    main()
