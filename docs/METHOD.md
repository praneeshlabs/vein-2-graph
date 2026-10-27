# Method, outputs, and validation

## Pipeline

1. Decode the input image with OpenCV.
2. Convert to grayscale and apply CLAHE for local contrast.
3. Create a dark-vein or bright-vein adaptive-threshold mask. Auto mode scores the two masks mainly by foreground coverage and prefers a sparse mask. This is a speed-oriented heuristic, not semantic vein recognition. The red/magenta option targets visibly red veins in HSV space.
4. Optionally estimate and mask the leaf silhouette to suppress background artifacts. This is skipped when no clear isolated blade boundary is found.
5. Remove tiny foreground components and optionally close small gaps. Large closing kernels can invent connections and cycles.
6. Skeletonize the mask, then optionally prune very short terminal spurs. Set spur length to 0 to disable this, since true short veins can otherwise be removed.
7. Cluster adjacent critical skeleton pixels into nodes and trace paths between nodes into a NetworkX MultiGraph. Endpoints and junctions are nodes. Closed degree-2 components receive an anchor. Parallel edges are retained.
8. Measure edge length by counting orthogonal steps as 1 pixel and diagonal steps as √2 pixels. Estimate width as twice the distance transform sampled along the skeleton.
9. Export JSON, node/edge CSVs, GraphML, masks, skeleton and overlay.

This is a classical CPU-first baseline inspired by the LEAF GUI workflow. The paper uses thresholding and cleaning, skeletonization, branchpoint separation, and a distance transform for width measurement. It warns that output quality depends on image quality and segmentation accuracy.

## Units

Lengths are in pixels by default. Enter pixels per mm only when calibrated from a known scale bar or documented image scale. Length in mm equals pixel length divided by pixels per mm. Do not infer a physical scale from PNG dimensions alone.

## Graph summary

V is node count, E is edge count, and C is connected component count. Independent cycles are E - V + C. This is the cycle rank, not every possible closed path. A disconnected network is not automatically an error; segmentation may have missed faint joins. Review the overlay.

## Accuracy limitations

- No universal threshold or polarity works for every species, acquisition method, illumination pattern, and vein stain.
- A clean-looking mask can still have false joins and missing tertiary veins. Compare the original image, mask, skeleton and graph overlay.
- The CNN paper's reported precision/recall harmonic mean of 94.5% ± 6% came from trained CNN ensembles and labelled data. That score must not be attributed to this thresholding baseline.
- For scientific validation, use manually labelled masks and report precision, recall, Dice/F1 and IoU, plus graph-topology checks. Review false junctions and missing vein segments manually.

## Multiple crops

Auto stitching uses OpenCV's feature-based stitcher and requires overlap and enough visual features. Grid mode combines same-sized crops in row-major order with user-specified overlap; it does not estimate rotation, perspective, or alignment. Inspect the reconstructed image because poorly aligned crops can create false edges and loops.

## Local checks

    python -m pip install -r requirements.txt
    python -m py_compile vein2graph.py app.py scripts/benchmark_images.py
    python -m pytest -q
    python vein2graph.py --input your_leaf.png --output-dir results
    streamlit run app.py
