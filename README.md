# Vein-2-Graph 🍃

A CPU-first pipeline and upload interface for turning leaf images into measurable vein graphs. It supports one complete leaf image and multiple cropped images, then exports a graph rather than only a binary mask.

**Important:** this is a classical image-processing baseline, not a trained CNN and not a guarantee that every vein is recovered. For scientific use, inspect the mask, skeleton and graph overlay and validate against expert-labelled masks. The first CNN paper's reported metric came from trained CNN ensembles and manually defined ground-truth regions; it does not apply to this code.

## Quick start

Use Python 3.10 or newer.

\`\`\`bash
python -m pip install -r requirements.txt
streamlit run app.py
\`\`\`

The browser interface lets you upload one full leaf or multiple crops. For crops, choose automatic stitching if images overlap, or grid reconstruction when their order and overlaps are known. Inspect the reconstructed image before trusting the graph.

For command-line processing:

\`\`\`bash
python vein2graph.py --input path/to/leaf.png --output-dir results
\`\`\`

Useful options include **--polarity auto/dark/bright/red**, **--block-size**, **--c-value**, **--min-component-size**, **--close-size**, **--spur-length**, and **--pixels-per-mm**. For close-up image crops without a visible full leaf outline, use **--no-leaf-mask**.

## Outputs

The output directory contains the enhanced grayscale image, binary vein mask, skeleton, graph overlay, JSON graph, node CSV, edge CSV, scalar-attribute GraphML and summary JSON. JSON edges include source/target node IDs and pixel paths. Lengths are in pixels unless calibrated from a known scale bar. Width is estimated as twice the distance transform sampled along the skeleton, following the LEAF GUI method.

Graph summary includes node/edge counts, connected components, endpoints, junctions, total edge length and independent cycles (E − V + C). Endpoints are nodes too. Adjacent junction pixels are grouped so a branchpoint is not counted as several separate nodes.

## Tests and image benchmark

\`\`\`bash
python -m py_compile vein2graph.py app.py scripts/benchmark_images.py
python -m pytest -q
python scripts/benchmark_images.py --input-dir data/images --output-dir benchmark_results
\`\`\`

For pixel-level metrics, provide same-sized reference masks named IMAGE_NAME_mask.png in a separate folder and pass the ground-truth directory option. White pixels in those masks must represent veins. The script calculates precision, recall, Dice/F1 and IoU. See docs/METHOD.md and docs/VALIDATION.md.

**Validation status:** automated graph/synthetic tests passed locally. Ten image-region smoke tests from the supplied paper figures ran successfully, but these are not ten independent Internet leaf photographs and are not enough to claim biological extraction accuracy. The test report documents this limitation.
