#!/usr/bin/env python3
"""CPU-first leaf-vein segmentation and graph extraction.

This is an inspectable classical baseline inspired by the thresholding,
cleaning, skeleton and distance-transform workflow in LEAF GUI (Price et al.,
2011). It does not claim perfect segmentation; inspect the mask/overlay.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))


def read_image(path: str | Path) -> np.ndarray:
    """Read common image formats, including paths containing non-ASCII chars."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def decode_upload(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image")
    return image


def _red_vein_mask(bgr: np.ndarray) -> np.ndarray:
    """Red-pixel candidate for leaves with visibly red/magenta vein staining."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b1 = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([12, 255, 255]))
    b2 = cv2.inRange(hsv, np.array([168, 45, 35]), np.array([180, 255, 255]))
    return cv2.bitwise_or(b1, b2) > 0


def _mask_quality(mask: np.ndarray) -> float:
    """Fast foreground-fraction heuristic for choosing adaptive-threshold polarity."""
    fraction = float(np.mean(mask))
    if fraction < 0.001 or fraction > 0.55:
        return -100.0 - abs(fraction - 0.08) * 100.0
    return -abs(math.log(max(fraction, 1e-6) / 0.08))


def estimate_leaf_region(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Estimate leaf silhouette; return None for crops without a clear blade boundary."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if min(h, w) < 40:
        return None
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.bitwise_not(bright)
    k = int(max(3, min(11, round(min(h, w) * 0.012))))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    candidates = []
    for binary in (bright, dark):
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), 8)
        for label in range(1, n):
            x, y, bw, bh, area = map(int, stats[label])
            fraction = area / float(h * w)
            touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            if touches or fraction < 0.04 or fraction > 0.92:
                continue
            comp = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
            filled_fraction = float(np.mean(filled > 0))
            if 0.04 <= filled_fraction <= 0.92:
                candidates.append((float(area), filled))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    region = candidates[0][1] > 0
    fraction = float(np.mean(region))
    return region if 0.04 <= fraction <= 0.92 else None


def segment_veins(
    bgr: np.ndarray,
    polarity: str = "auto",
    block_size: int = 41,
    c_value: int = 7,
    min_component_size: int = 10,
    close_size: int = 1,
    clahe_clip: float = 2.0,
    constrain_to_leaf: bool = True,
    leaf_region_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return enhanced grayscale, uint8 white-vein mask, and chosen polarity."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1
    if polarity == "red":
        raw, chosen = _red_vein_mask(bgr), "red"
    else:
        # OpenCV uses local_mean - C, so bright ridges require a negative C.
        dark = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, int(c_value)) > 0
        bright = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, -int(c_value)) > 0
        if polarity == "dark":
            raw, chosen = dark, "dark"
        elif polarity == "bright":
            raw, chosen = bright, "bright"
        elif polarity == "auto":
            raw, chosen = (dark, "dark") if _mask_quality(dark) >= _mask_quality(bright) else (bright, "bright")
        else:
            raise ValueError("polarity must be one of: auto, dark, bright, red")
    if min_component_size > 1:
        raw = remove_small_objects(raw.astype(bool), max_size=max(0, int(min_component_size) - 1), connectivity=2)
    leaf_region = leaf_region_mask
    if leaf_region is None and constrain_to_leaf:
        leaf_region = estimate_leaf_region(bgr)
    if leaf_region is not None:
        raw &= leaf_region
    if int(close_size) > 1:
        k = int(close_size) + (int(close_size) % 2 == 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        raw = cv2.morphologyEx(raw.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0
    return enhanced, raw.astype(np.uint8) * 255, chosen


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask > 0)


def _neighbors(skel: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    h, w = skel.shape
    out = []
    for dy, dx in OFFSETS:
        yy, xx = y + dy, x + dx
        if not (0 <= yy < h and 0 <= xx < w and skel[yy, xx]):
            continue
        # Suppress diagonal shortcuts when an orthogonal link already connects this neighborhood.
        if dy != 0 and dx != 0 and (skel[y, xx] or skel[yy, x]):
            continue
        out.append((yy, xx))
    return out


def prune_short_spurs(skeleton: np.ndarray, max_length: int = 2) -> np.ndarray:
    """Remove very short endpoint paths ending at a junction; 0 disables pruning."""
    result = np.asarray(skeleton, dtype=bool).copy()
    max_length = max(0, int(max_length))
    if max_length == 0 or not result.any():
        return result
    for _ in range(3):
        changed = False
        endpoints = [tuple(map(int, p)) for p in np.argwhere(result)]
        for endpoint in endpoints:
            if not result[endpoint] or len(_neighbors(result, *endpoint)) != 1:
                continue
            path, previous, current = [endpoint], None, endpoint
            while len(path) <= max_length:
                options = [q for q in _neighbors(result, *current) if q != previous]
                if len(options) != 1:
                    break
                nxt = options[0]
                nxt_degree = len(_neighbors(result, *nxt))
                if nxt_degree >= 3:
                    for p in path:
                        result[p] = False
                    changed = True
                    break
                if nxt_degree != 2:
                    break
                path.append(nxt)
                previous, current = current, nxt
        if not changed:
            break
    return result


def extract_graph(skeleton: np.ndarray, distance_map: Optional[np.ndarray] = None,
                  pixels_per_mm: Optional[float] = None) -> nx.MultiGraph:
    """Convert a skeleton to a MultiGraph while avoiding per-pixel neighbor dictionaries."""
    skel = np.asarray(skeleton, dtype=bool)
    if skel.ndim != 2:
        raise ValueError("skeleton must be a 2D array")
    h, w = skel.shape
    graph = nx.MultiGraph()
    graph.graph.update(width_px=w, height_px=h, pixels_per_mm=pixels_per_mm,
                       length_unit="mm" if pixels_per_mm else "px")
    coords = np.argwhere(skel)
    if coords.size == 0:
        return graph
    critical: set[tuple[int, int]] = set()
    for y0, x0 in coords:
        y, x = int(y0), int(x0)
        if len(_neighbors(skel, y, x)) != 2:
            critical.add((y, x))

    # Pure degree-2 connected loops need an anchor node to remain representable.
    n_components, component_labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    component_has_critical = np.zeros(n_components, dtype=bool)
    for y, x in critical:
        component_has_critical[int(component_labels[y, x])] = True
    pixel_component_ids = None
    if not np.all(component_has_critical[1:]):
        pixel_component_ids = component_labels[coords[:, 0], coords[:, 1]]
        for component_id in range(1, n_components):
            if not component_has_critical[component_id]:
                first = np.flatnonzero(pixel_component_ids == component_id)
                if first.size:
                    y, x = coords[int(first[0])]
                    critical.add((int(y), int(x)))
    del component_labels, component_has_critical, pixel_component_ids

    # Merge adjacent critical pixels into one logical node.
    remaining = set(critical)
    groups: list[set[tuple[int, int]]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group, stack = {seed}, [seed]
        while stack:
            p = stack.pop()
            for q in _neighbors(skel, p[0], p[1]):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)

    node_for: dict[tuple[int, int], int] = {}
    for node_id, group in enumerate(groups, start=1):
        node_for.update((p, node_id) for p in group)
        xs, ys = [p[1] for p in group], [p[0] for p in group]
        arms = set()
        for p in group:
            for q in _neighbors(skel, *p):
                if q not in group:
                    arms.add((p, q) if p <= q else (q, p))
        arm_count = len(arms)
        kind = "junction" if arm_count >= 3 else "endpoint" if arm_count == 1 else "isolated" if arm_count == 0 else "transition"
        graph.add_node(node_id, x=float(np.mean(xs)), y=float(np.mean(ys)),
                       pixel_count=len(group), kind=kind, arm_count=arm_count)

    # Direction bits in a uint8 image mark visited links without storing tuple-pairs.
    bit_for = {offset: (1 << idx) for idx, offset in enumerate(OFFSETS)}
    visited_dirs = np.zeros((h, w), dtype=np.uint8)
    def link_visited(a, b):
        dy, dx = b[0] - a[0], b[1] - a[1]
        return bool(visited_dirs[a[0], a[1]] & bit_for[(dy, dx)])
    def mark_link(a, b):
        dy, dx = b[0] - a[0], b[1] - a[1]
        visited_dirs[a[0], a[1]] |= bit_for[(dy, dx)]
        visited_dirs[b[0], b[1]] |= bit_for[(-dy, -dx)]

    edge_id = 0
    for node_id, group in enumerate(groups, start=1):
        for start_pixel in group:
            for first_next in _neighbors(skel, *start_pixel):
                if first_next in group or link_visited(start_pixel, first_next):
                    continue
                mark_link(start_pixel, first_next)
                path = [start_pixel, first_next]
                previous, current = start_pixel, first_next
                target, guard = node_for.get(current), 0
                while target is None and guard <= int(coords.shape[0]) + 1:
                    guard += 1
                    candidates = [q for q in _neighbors(skel, *current) if q != previous]
                    if not candidates:
                        break
                    next_pixel = next((q for q in candidates if not link_visited(current, q)), None)
                    if next_pixel is None:
                        break
                    mark_link(current, next_pixel)
                    path.append(next_pixel)
                    previous, current = current, next_pixel
                    target = node_for.get(current)
                if target is None:
                    continue
                path_array = np.asarray(path, dtype=np.int32)
                diffs = np.diff(path_array, axis=0)
                length_px = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum()) if len(path) > 1 else 0.0
                if distance_map is not None and len(path):
                    mean_width_px = float((2.0 * distance_map[path_array[:, 0], path_array[:, 1]]).mean())
                else:
                    mean_width_px = None
                length = length_px / pixels_per_mm if pixels_per_mm else length_px
                graph.add_edge(node_id, target, key=edge_id, edge_id=edge_id,
                               length_px=length_px, length=float(length),
                               length_unit="mm" if pixels_per_mm else "px",
                               mean_width_px=mean_width_px,
                               mean_width=(mean_width_px / pixels_per_mm if mean_width_px is not None and pixels_per_mm else None),
                               width_unit="mm" if pixels_per_mm else "px",
                               path_rc=[[int(y), int(x)] for y, x in path])
                edge_id += 1
    return graph


def graph_summary(graph: nx.MultiGraph) -> dict:
    v, e = graph.number_of_nodes(), graph.number_of_edges()
    components = nx.number_connected_components(graph) if v else 0
    degree = dict(graph.degree())
    kinds = {}
    for _, d in graph.nodes(data=True):
        kinds[d.get("kind", "unknown")] = kinds.get(d.get("kind", "unknown"), 0) + 1
    lengths = [float(d.get("length", d.get("length_px", 0.0))) for *_, d in graph.edges(data=True, keys=True)]
    return {
        "nodes": v, "edges": e, "connected_components": components,
        "connected": bool(v > 0 and components == 1),
        "independent_cycles": e - v + components,
        "endpoints": kinds.get("endpoint", 0), "junctions": kinds.get("junction", 0),
        "isolated_nodes": kinds.get("isolated", 0), "transition_nodes": kinds.get("transition", 0),
        "mean_degree": (float(np.mean(list(degree.values()))) if degree else 0.0),
        "total_length": float(sum(lengths)),
        "length_unit": graph.graph.get("length_unit", "px"),
    }


def make_overlay(bgr: np.ndarray, mask: np.ndarray, skeleton: np.ndarray, graph: nx.MultiGraph) -> np.ndarray:
    out = bgr.copy()
    green = np.zeros_like(out); green[:, :, 1] = 210
    idx = mask > 0
    out[idx] = cv2.addWeighted(out[idx], 0.58, green[idx], 0.42, 0)
    out[skeleton] = (0, 0, 255)
    for _, data in graph.nodes(data=True):
        cv2.circle(out, (int(round(data["x"])), int(round(data["y"]))), 3, (255, 70, 0), -1, cv2.LINE_AA)
    return out


def _serializable_graph(graph: nx.MultiGraph, summary: dict) -> dict:
    nodes = [{"id": int(n), **d} for n, d in graph.nodes(data=True)]
    edges = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        edges.append({"id": int(d["edge_id"]), "source": int(u), "target": int(v), **d})
    return {"schema": "vein-2-graph/v1", "coordinate_system": "pixel coordinates; x right, y down; origin top-left",
            "graph_metadata": graph.graph, "summary": summary, "nodes": nodes, "edges": edges}


def process_image(bgr: np.ndarray, polarity: str = "auto", block_size: int = 41,
                  c_value: int = 7, min_component_size: int = 10, close_size: int = 1,
                  pixels_per_mm: Optional[float] = None, constrain_to_leaf: bool = True,
                  spur_length: int = 2) -> dict:
    if pixels_per_mm is not None and pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be > 0")
    t0 = time.perf_counter()
    leaf_region = estimate_leaf_region(bgr) if constrain_to_leaf else None
    gray, mask, used_polarity = segment_veins(bgr, polarity, block_size, c_value,
                                               min_component_size, close_size,
                                               constrain_to_leaf=constrain_to_leaf, leaf_region_mask=leaf_region)
    skeleton = prune_short_spurs(skeletonize_mask(mask), spur_length)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    graph = extract_graph(skeleton, distance, pixels_per_mm)
    summary = graph_summary(graph)
    overlay = make_overlay(bgr, mask, skeleton, graph)
    return {"gray": gray, "mask": mask,
            "leaf_region": (leaf_region.astype(np.uint8) * 255 if leaf_region is not None else None),
            "skeleton": skeleton, "distance": distance, "graph": graph, "summary": summary,
            "overlay": overlay, "polarity_used": used_polarity,
            "elapsed_seconds": time.perf_counter() - t0}


def save_outputs(result: dict, output_dir: str | Path, source_name: str = "leaf") -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{source_name}_enhanced.png"), result["gray"])
    cv2.imwrite(str(out / f"{source_name}_mask.png"), result["mask"])
    if result.get("leaf_region") is not None:
        cv2.imwrite(str(out / f"{source_name}_leaf_region.png"), result["leaf_region"])
    cv2.imwrite(str(out / f"{source_name}_skeleton.png"), result["skeleton"].astype(np.uint8) * 255)
    cv2.imwrite(str(out / f"{source_name}_overlay.png"), result["overlay"])
    graph = result["graph"]
    (out / f"{source_name}_graph.json").write_text(json.dumps(_serializable_graph(graph, result["summary"]), indent=2), encoding="utf-8")
    with (out / f"{source_name}_nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "x", "y", "kind", "arm_count", "pixel_count"]); writer.writeheader()
        for n, d in graph.nodes(data=True):
            writer.writerow({"node_id": n, **{k: d.get(k) for k in writer.fieldnames[1:]}})
    with (out / f"{source_name}_edges.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["edge_id", "source", "target", "length_px", "length", "length_unit", "mean_width_px", "mean_width", "width_unit"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for u, v, k, d in graph.edges(keys=True, data=True):
            writer.writerow({"source": u, "target": v, **{key: d.get(key) for key in fields if key not in ("source", "target")}})
    simple = nx.MultiGraph()
    for n, d in graph.nodes(data=True):
        simple.add_node(n, x=float(d["x"]), y=float(d["y"]), kind=str(d["kind"]), arm_count=int(d["arm_count"]))
    for u, v, k, d in graph.edges(keys=True, data=True):
        simple.add_edge(u, v, key=int(k), edge_id=int(d["edge_id"]), length=float(d["length"]),
                        length_unit=str(d["length_unit"]), length_px=float(d["length_px"]))
    nx.write_graphml(simple, out / f"{source_name}_graph.graphml")
    (out / f"{source_name}_summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")


def stitch_images(images: Sequence[np.ndarray], mode: str = "auto", rows: int = 1, cols: int = 1,
                  overlap_x: int = 0, overlap_y: int = 0) -> tuple[np.ndarray, str]:
    """Stitch overlapping crops automatically or place them in a known grid."""
    if not images:
        raise ValueError("Upload at least one image")
    if len(images) == 1:
        return images[0], "single image; no stitching needed"
    if mode == "auto":
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched = stitcher.stitch(list(images))
        if status != cv2.Stitcher_OK or stitched is None:
            raise ValueError(f"Automatic stitching failed (OpenCV status {status}). Try grid mode or provide better-overlapped crops.")
        return stitched, "OpenCV feature-based stitching"
    if mode != "grid":
        raise ValueError("mode must be auto or grid")
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1 or rows * cols != len(images):
        raise ValueError("For grid stitching, rows × columns must equal number of crops")
    h, w = images[0].shape[:2]
    if any(im.shape[:2] != (h, w) for im in images):
        raise ValueError("Grid mode requires crops of identical dimensions")
    if not (0 <= overlap_x < w and 0 <= overlap_y < h):
        raise ValueError("Overlap must be non-negative and smaller than crop dimensions")
    step_x, step_y = w - overlap_x, h - overlap_y
    canvas = np.zeros((step_y * (rows - 1) + h, step_x * (cols - 1) + w, 3), np.float32)
    weights = np.zeros(canvas.shape[:2], np.float32)
    for i, im in enumerate(images):
        r, c = divmod(i, cols); y, x = r * step_y, c * step_x
        yy = np.ones(h, np.float32); xx = np.ones(w, np.float32)
        if overlap_y:
            ramp = np.linspace(0.05, 1.0, overlap_y, dtype=np.float32)
            if r > 0: yy[:overlap_y] = ramp
            if r < rows - 1: yy[-overlap_y:] = ramp[::-1]
        if overlap_x:
            ramp = np.linspace(0.05, 1.0, overlap_x, dtype=np.float32)
            if c > 0: xx[:overlap_x] = ramp
            if c < cols - 1: xx[-overlap_x:] = ramp[::-1]
        wt = yy[:, None] * xx[None, :]
        canvas[y:y+h, x:x+w] += im.astype(np.float32) * wt[:, :, None]
        weights[y:y+h, x:x+w] += wt
    out = canvas / np.maximum(weights[:, :, None], 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8), "known grid placement with overlap blending (no alignment estimation)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract an approximate leaf-vein graph from an image")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--polarity", choices=["auto", "dark", "bright", "red"], default="auto")
    parser.add_argument("--block-size", type=int, default=41)
    parser.add_argument("--c-value", type=int, default=7)
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--close-size", type=int, default=1)
    parser.add_argument("--pixels-per-mm", type=float, default=None)
    parser.add_argument("--spur-length", type=int, default=2)
    parser.add_argument("--no-leaf-mask", action="store_true")
    args = parser.parse_args(argv)
    image = read_image(args.input)
    result = process_image(image, args.polarity, args.block_size, args.c_value,
                           args.min_component_size, args.close_size, args.pixels_per_mm,
                           constrain_to_leaf=not args.no_leaf_mask, spur_length=args.spur_length)
    save_outputs(result, args.output_dir)
    print(json.dumps({"elapsed_seconds": round(result["elapsed_seconds"], 3),
                      "polarity_used": result["polarity_used"], **result["summary"],
                      "output_dir": str(Path(args.output_dir).resolve())}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
