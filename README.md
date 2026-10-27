# Vein-2-Graph

CPU-first toolkit for extracting leaf-vein graphs from whole-leaf images. It includes image preprocessing, vein segmentation, skeletonization, graph construction, a Streamlit upload interface, and synthetic tests.

> **Scientific limitation:** classic thresholding is a baseline, not a guarantee of perfect extraction. Inspect masks and overlays. Validating segmentation accuracy requires ground-truth vein masks for representative real leaves.

## Quick start

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Or use the command line:

```bash
python vein2graph.py --input path/to/leaf.png --output-dir results
```

For processing details, outputs, tuning controls, and testing, see `docs/METHOD.md`.
