from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from vein2graph import decode_upload, process_image, save_outputs, stitch_images, _serializable_graph

st.set_page_config(page_title="Vein-2-Graph", page_icon="🍃", layout="wide")
st.title("🍃 Vein-2-Graph")
st.caption("Upload a leaf image and export a graph. CPU-only classical image processing.")
st.warning("This is a baseline, not a trained CNN or a guarantee that every vein is found. Inspect the mask and overlay before using graph statistics scientifically.")

mode = st.radio("Input mode", ["One complete leaf image", "Multiple cropped images"], horizontal=True)
if mode == "One complete leaf image":
    up = st.file_uploader("Upload a complete leaf image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    uploads = [up] if up else []
else:
    uploads = st.file_uploader("Upload crops from the same leaf", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True) or []

st.sidebar.header("Segmentation")
polarity_label = st.sidebar.selectbox("Vein appearance", [
    "Auto choose", "Dark veins on lighter background", "Bright veins on darker background", "Red/magenta veins"
])
polarity = {
    "Auto choose": "auto",
    "Dark veins on lighter background": "dark",
    "Bright veins on darker background": "bright",
    "Red/magenta veins": "red",
}[polarity_label]
block_size = st.sidebar.slider("Adaptive threshold window", 15, 101, 41, 2)
c_value = st.sidebar.slider("Threshold offset", 0, 20, 7)
min_component_size = st.sidebar.slider("Remove tiny components (pixels)", 0, 100, 10)
close_size = st.sidebar.select_slider("Connect tiny gaps", options=[1, 3, 5], value=1,
                                      help="1 disables closing. Larger kernels can create false connections.")
spur_length = st.sidebar.slider("Prune terminal spurs (pixels)", 0, 8, 2,
                                 help="Set 0 to preserve every terminal branch. Pruning may remove genuine tiny veins.")
pixels_per_mm = st.sidebar.number_input("Pixels per mm (optional)", min_value=0.0, value=0.0, step=10.0,
                                        help="Enter a measured scale-bar calibration. Leave 0 for pixel units.")
constrain_to_leaf = st.sidebar.checkbox(
    "Mask background outside the detected leaf boundary",
    value=(mode == "One complete leaf image"),
    key=f"leaf_boundary_{mode}",
    help="Recommended for full-leaf images. Disable for close-up crops without a visible blade outline."
)

stitch_mode, grid_rows, grid_cols, overlap_x, overlap_y = "auto", 1, 1, 0, 0
if mode == "Multiple cropped images":
    choice = st.radio("Crop reconstruction", ["Overlapping crops, auto-stitch", "Known grid layout"], horizontal=True)
    if choice == "Known grid layout":
        stitch_mode = "grid"
        a, b, c, d = st.columns(4)
        grid_rows = a.number_input("Rows", min_value=1, value=1, step=1)
        grid_cols = b.number_input("Columns", min_value=1, value=max(1, len(uploads)), step=1)
        overlap_x = c.number_input("Horizontal overlap (px)", min_value=0, value=0, step=10)
        overlap_y = d.number_input("Vertical overlap (px)", min_value=0, value=0, step=10)
        st.caption("Grid mode assumes row-major upload order, equal crop dimensions and known overlap. It does not estimate alignment.")
    else:
        st.caption("Auto-stitch requires overlap and enough visual features. If it fails, try known grid layout.")

if uploads:
    try:
        images = [decode_upload(f.getvalue()) for f in uploads]
        if mode == "One complete leaf image":
            bgr, stitch_note = images[0], "Single complete image"
        else:
            bgr, stitch_note = stitch_images(
                images, mode=stitch_mode, rows=int(grid_rows), cols=int(grid_cols),
                overlap_x=int(overlap_x), overlap_y=int(overlap_y)
            )
        st.caption(f"{len(images)} file(s); working image: {bgr.shape[1]:,} × {bgr.shape[0]:,} pixels. {stitch_note}.")
        if st.button("Extract vein graph", type="primary"):
            with st.spinner("Segmenting veins and tracing the graph on CPU..."):
                result = process_image(
                    bgr, polarity=polarity, block_size=block_size, c_value=c_value,
                    min_component_size=min_component_size, close_size=close_size,
                    pixels_per_mm=pixels_per_mm if pixels_per_mm > 0 else None,
                    constrain_to_leaf=constrain_to_leaf, spur_length=spur_length
                )
            stats, graph = result["summary"], result["graph"]
            st.success(f"Finished in {result['elapsed_seconds']:.2f} s. Polarity: {result['polarity_used']}.")
            m = st.columns(5)
            m[0].metric("Nodes", f"{stats['nodes']:,}")
            m[1].metric("Edges", f"{stats['edges']:,}")
            m[2].metric("Independent cycles", f"{stats['independent_cycles']:,}")
            m[3].metric("Components", f"{stats['connected_components']:,}")
            m[4].metric("Connected?", "Yes" if stats["connected"] else "No")
            c1, c2, c3 = st.columns(3)
            c1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input / reconstructed image", use_container_width=True)
            c2.image(result["mask"], caption="Binary vein mask", clamp=True, use_container_width=True)
            c3.image(cv2.cvtColor(result["overlay"], cv2.COLOR_BGR2RGB), caption="Graph overlay: red skeleton, blue nodes", use_container_width=True)
            with st.expander("Skeleton and full summary"):
                st.image(result["skeleton"].astype(np.uint8) * 255, caption="Skeleton", clamp=True, use_container_width=True)
                st.json(stats)
            json_bytes = json.dumps(_serializable_graph(graph, stats), indent=2).encode()
            st.download_button("Download graph JSON", json_bytes, file_name="leaf_vein_graph.json", mime="application/json")
            nodes_csv = io.StringIO()
            nodes_csv.write("node_id,x,y,kind,arm_count,pixel_count\n")
            for n, data in graph.nodes(data=True):
                nodes_csv.write(f"{n},{data['x']},{data['y']},{data['kind']},{data['arm_count']},{data['pixel_count']}\n")
            edges_csv = io.StringIO()
            edges_csv.write("edge_id,source,target,length_px,length,mean_width_px,mean_width\n")
            for u, v, k, data in graph.edges(keys=True, data=True):
                edges_csv.write(f"{data['edge_id']},{u},{v},{data['length_px']},{data['length']},{data['mean_width_px']},{data['mean_width']}\n")
            d1, d2 = st.columns(2)
            d1.download_button("Nodes CSV", nodes_csv.getvalue().encode(), file_name="leaf_nodes.csv", mime="text/csv")
            d2.download_button("Edges CSV", edges_csv.getvalue().encode(), file_name="leaf_edges.csv", mime="text/csv")
            ok, png = cv2.imencode(".png", result["overlay"])
            if ok:
                st.download_button("Graph overlay PNG", png.tobytes(), file_name="leaf_graph_overlay.png", mime="image/png")
            with tempfile.TemporaryDirectory() as td:
                save_outputs(result, td)
                bundle = io.BytesIO()
                with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
                    for path in Path(td).iterdir():
                        zf.write(path, arcname=path.name)
                zip_bytes = bundle.getvalue()
            st.download_button("Download all outputs (.zip)", zip_bytes, file_name="vein2graph_outputs.zip", mime="application/zip")
            st.info("Independent cycles are E − V + C. Endpoints are nodes too. Graph counts describe the extracted mask, not guaranteed biological ground truth.")
    except Exception as exc:
        st.error(f"Could not process these images: {exc}")
else:
    st.info("Upload a full leaf image or select multiple-crop mode to begin.")

with st.expander("Tips"):
    st.write("Use a clear, high-resolution scan or backlit image. Try Auto first, then compare dark/bright/red polarity if veins are missed. Keep closing kernels small. Multi-crop stitching needs overlap or a known grid. Use a scale bar to report millimetres; otherwise the graph uses pixels.")
