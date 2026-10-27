import cv2
import numpy as np
from vein2graph import process_image, stitch_images


def synthetic_leaf(seed=0, size=256):
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 24, np.uint8)
    pts = np.array([[size//2, 20], [size-25, size//2], [size//2, size-20], [25, size//2]], np.int32)
    cv2.fillPoly(image, [pts], (125, 165, 90))
    center = (size//2, size//2)
    cv2.line(image, (center[0], 20), (center[0], size-20), (225, 235, 215), 4)
    for y in range(45, size-40, 18):
        offset = int((y-size//2) * 0.7)
        cv2.line(image, (center[0], y), (max(28, center[0]-offset), min(size-22, y-15)), (205, 218, 180), 2)
        cv2.line(image, (center[0], y), (min(size-28, center[0]+offset), min(size-22, y+15)), (205, 218, 180), 2)
    noise = rng.normal(0, 2, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_pipeline_outputs_nonempty_graph_on_synthetic_leaf():
    result = process_image(synthetic_leaf(), polarity="bright", block_size=31, min_component_size=3)
    assert result["mask"].shape[:2] == (256, 256)
    assert result["skeleton"].any()
    assert result["graph"].number_of_nodes() > 0
    assert result["overlay"].shape == (256, 256, 3)


def test_grid_stitching_places_tiles_with_expected_size():
    a = np.zeros((20, 30, 3), np.uint8)
    b = np.full((20, 30, 3), 255, np.uint8)
    merged, note = stitch_images([a, b], mode="grid", rows=1, cols=2, overlap_x=5)
    assert merged.shape == (20, 55, 3)
    assert "grid" in note
