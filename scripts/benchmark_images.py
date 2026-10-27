#!/usr/bin/env python3
"""Benchmark Vein-2-Graph over a directory of images.

Optional ground-truth masks must be same-sized, with white pixels representing
veins. Reference masks are matched as <image-stem>_mask.png.
"""
from __future__ import annotations
import argparse, csv, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from vein2graph import process_image, read_image, save_outputs

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def mask_metrics(pred: np.ndarray, truth_path: Path) -> dict:
    truth_img = cv2.imread(str(truth_path), cv2.IMREAD_GRAYSCALE)
    if truth_img is None:
        raise ValueError(f"Could not read ground-truth mask: {truth_path}")
    if truth_img.shape != pred.shape:
        raise ValueError(f"Mask shape mismatch for {truth_path}: {truth_img.shape} vs {pred.shape}")
    p, g = pred > 0, truth_img > 0
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = int(np.count_nonzero(p | g))
    return {"precision": precision, "recall": recall, "f1_dice": f1,
            "iou": tp / union if union else 1.0, "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder containing test images")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--ground-truth-dir", help="Folder with <stem>_mask.png reference masks")
    parser.add_argument("--polarity", choices=["auto", "dark", "bright", "red"], default="auto")
    parser.add_argument("--block-size", type=int, default=41)
    parser.add_argument("--c-value", type=int, default=7)
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--close-size", type=int, default=1)
    parser.add_argument("--spur-length", type=int, default=2)
    parser.add_argument("--pixels-per-mm", type=float)
    parser.add_argument("--no-leaf-mask", action="store_true")
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        parser.error(f"No supported images found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(files, 1):
        start = time.perf_counter()
        image = read_image(path)
        result = process_image(
            image, polarity=args.polarity, block_size=args.block_size, c_value=args.c_value,
            min_component_size=args.min_component_size, close_size=args.close_size,
            pixels_per_mm=args.pixels_per_mm, constrain_to_leaf=not args.no_leaf_mask,
            spur_length=args.spur_length
        )
        save_outputs(result, output_dir / path.stem, source_name=path.stem)
        stats = result["summary"]
        row = {
            "image": path.name, "width_px": image.shape[1], "height_px": image.shape[0],
            "seconds": round(time.perf_counter() - start, 4), "polarity_used": result["polarity_used"],
            "mask_fraction": float(np.mean(result["mask"] > 0)),
            "skeleton_pixels": int(result["skeleton"].sum()), **stats
        }
        if args.ground_truth_dir:
            gt = Path(args.ground_truth_dir) / f"{path.stem}_mask.png"
            if not gt.exists():
                row.update({"ground_truth_mask": str(gt), "metric_error": "mask file not found"})
            else:
                try:
                    row.update(mask_metrics(result["mask"], gt))
                    row["ground_truth_mask"] = str(gt)
                except ValueError as exc:
                    row["metric_error"] = str(exc)
        rows.append(row)
        print(f"[{i}/{len(files)}] {path.name}: {stats['nodes']} nodes, {stats['edges']} edges, {row['seconds']:.3f}s")
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nProcessed {len(rows)} images. Report: {(output_dir / 'benchmark.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
