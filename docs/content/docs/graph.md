---
title: "Crack Graph"
weight: 8
---

The `combra.graph` module turns a binarised crack image into a directed graph (`networkx.DiGraph`) whose nodes are contour vertices and whose edges are short straight segments classified as Co, WC-Co, WC, or WC-WC. From there you can run shortest-path / minimum-energy-path searches.

Edge type encoding:
- `0` — Co
- `1` — WC-Co
- `2` — WC
- `3` — WC-WC

## Build

```python
from combra import graph, data

img = data.crack_images()[0][1]

(entry_nodes, exit_nodes,
 img_contours_o, img_preprocessed,
 cnts, nodes_metadata) = graph.preprocess_graph_image(
    img, border=30, disk=5, entry_ellps_w=5, exit_ellps_w=5, r=4)

g, img_contours = graph.create_crack_graph(
    img_preprocessed.shape, cnts, nodes_metadata, eps=300)
```

### `preprocess_graph_image(image, r=2, border=30, border_node_eps=10, tol=5, disk=5, labeled_cnts=False, labels=False, entry_ellps_w=1, exit_ellps_w=1)`
Median-filter, Otsu, and contour-extract the input. Returns `(entry_nodes, exit_nodes, img_contours_o, img_preprocessed, cnts, nodes_metadata)`. Pass `labeled_cnts=True` (and `labels`) when you have hand-annotated contours and want to skip the binarisation.

### `create_crack_graph(img_shape, cnts, nodes_metadata, eps=100, line_eps=3, border=30, border_eps=0, border_number_min=2, border_pixel=255, same_node_eps=5, labels=False, labeled_line_eps=10, workers=10)`
Build the graph. `eps` is the maximum edge length in pixels; `line_eps` is the perpendicular tolerance used when classifying edges. Returns `(g, img_contours)`.

### `get_edges(start_node_index, nodes_metadata, process_metadata)`
Compute outgoing edges from one node — used internally by `create_crack_graph`. You normally don't need to call it directly.

### `get_edge_type(node1, node2, cnts, nodes_metadata, wc_eps=30, border_pixel=0)`
Classify a single edge between two node indices into Co / WC-Co / WC / WC-WC.

### `get_edge_type_labeled(node1, node2, nodes_metadata, line_eps=10)`
Same as above but uses hand labels carried in `nodes_metadata`.

## Energies and paths

```python
import numpy as np

energy_conf = np.zeros((20, 20)).tolist()
for i, en1 in enumerate(range(20)):       # Co
    for j, en2 in enumerate(range(20)):   # WC-Co
        energy_conf[i][j] = {0: en1, 1: en2, 2: 20, 3: 0}

energies_paths = graph.get_energies(
    energy_conf, g, cnts, nodes_metadata,
    entry_nodes=[0, 1, 3], exit_nodes=[63, 64, 67],
    first_k_paths=1, parallel=True, workers=20)
```

### `get_energies(energy_conf, g, cnts, nodes_metadata, entry_nodes, exit_nodes, first_k_paths=2, parallel=False, workers=23, recalculate_paths=False)`
Sweep an `(N, M)` grid of edge-type weights. For every grid cell, set the edge weights and run k-shortest-path between every (entry, exit) pair. Returns a nested list aligned with `energy_conf`.

### `find_shortest_energy_paths(G, cnts, nodes_metadata, entry_node, exit_node, k, recalculate_paths=False)`
Find the `k` shortest paths between one entry/exit pair and return a `pandas.DataFrame` with per-path lengths, energies, and per-edge-type breakdowns (Co/WC/WC-Co counts and pixel fractions).

### `fixed_paths_energies(g, cnts, nodes_metadata, entry_nodes, exit_nodes, workers=23, ...)`
Compute energies along a fixed set of paths (no optimisation).

### `paths_queues()`
Internal queue-based path enumerator (no public arguments).

## Plot

### `graph_plot(g, img_contours=None, name='graph.html', save=False, node_size=12, edge_width=2, color_dict=None, edge_width_dict=None)`
Interactive Plotly plot of the graph overlaid on `img_contours`.

### `plot_optimized_energies(energies_paths, path_index=0, N=5, M=5, y_label='co_e', x_label='wc-co_e', fontsize_h=10, fontsize_axes=50)`
Heatmap of optimal path energies over the `(Co, WC-Co)` weight grid for path `path_index`.

### `plot_paths(g, df, img_aligned, border=30)`
Overlay the paths in `df` (output of `find_shortest_energy_paths`) on the background image.

### `plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10)`
Overlay the energy-optimised paths from `get_energies` on the contour image.

### `draw_tree(img, centres=False, leafs=False, nodes=False, bones=False)`
Render skeleton landmarks (centres / leaves / nodes / skeleton pixels) onto a binary image.
