# Validation notes

## Automated tests

The test suite covers an empty skeleton, a straight line, a T-junction, a closed square loop, disconnected lines, a synthetic leaf image, grid reconstruction and terminal-spur pruning. Run python -m pytest -q.

## Ten image-region smoke test

During development, ten image regions were cropped from example figures embedded in the supplied LEAF GUI paper (Price et al., 2011): a source leaf photo, the paper's thresholded vein illustration, three leaf-figure panels, two screenshot leaf panels and three venation close-ups. They are local development crops, not ten independently sourced Internet leaf photographs. Several are already-processed figure panels, not raw photographs. They exercise the pipeline on contrasting image types but are **not a scientific accuracy benchmark**.

The ten crops were processed and exported. Including file exports, observed runtime was approximately 0.025–0.15 seconds per small region, with a mean near 0.084 seconds on this runtime. These images were only about 140–395 pixels on a side; do not extrapolate the timing to multi-megapixel scans.

The raw-leaf figure crop produced a usable coarse skeleton, but some branches were disconnected and some regions were over-segmented. The displayed segmented panel is not pixel-aligned ground truth, so direct pixel-level F1/IoU would be misleading without registration and expert annotations. Textured or already-processed figure panels generated dense graphs, an expected failure mode for thresholding. These tests do **not** establish that every biological vein is recovered.

## Real accuracy benchmark

Prepare at least ten representative raw leaf images and expert-annotated vein masks at exactly the same dimensions. White should mean vein and black background. Run:

    python scripts/benchmark_images.py --input-dir data/images --ground-truth-dir data/masks --output-dir benchmark_results

The script writes CSV and JSON reports with runtime, graph statistics, precision, recall, Dice/F1 and IoU when matching masks exist. Use multiple species and imaging conditions. Inspect overlays manually, especially junctions and fine tertiary veins. The development runtime had no external image-host access, so I did not claim to download or validate ten new Internet images.
