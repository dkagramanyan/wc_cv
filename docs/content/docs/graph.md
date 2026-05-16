---
title: "Crack Graph"
weight: 8
---

The `combra.graph` module turns a binarised crack image into a directed graph (`networkx.DiGraph`) whose nodes are contour vertices and whose edges are short straight segments classified as Co, WC-Co, WC, or WC-WC. From there you can run shortest-path / minimum-energy-path searches.

```python
from combra import graph
```

Edge type encoding:

| value | meaning |
| --- | --- |
| `0` | Co |
| `1` | WC-Co |
| `2` | WC |
| `3` | WC-WC |

## Build

### `combra.graph.preprocess_graph_image`

```python
preprocess_graph_image(image, r=2, border=30, border_node_eps=10, tol=5, disk=5,
                       labeled_cnts=False, labels=False,
                       entry_ellps_w=1, exit_ellps_w=1)
```

Median-filter, Otsu, and contour-extract the input. Pass `labeled_cnts=True` (and `labels`) when you have hand-annotated contours and want to skip binarisation.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `image` | `ndarray` | — | Source image (any channels). |
| `r` | `int` | `2` | Marker radius for node visualisation. |
| `border` | `int` | `30` | Padding added before extraction. |
| `border_node_eps` | `int` | `10` | Max distance from border for a node to count as an entry/exit candidate. |
| `tol` | `float` | `5` | Douglas–Peucker tolerance. |
| `disk` | `int` | `5` | Median-filter footprint radius. |
| `labeled_cnts` | `bool` | `False` | Skip binarisation; use `labels` directly. |
| `labels` | `bool` | `False` | Hand-labelled contour data. |
| `entry_ellps_w`, `exit_ellps_w` | `int` | `1` | Width of the entry/exit ellipse region overlay. |

**Returns**

| name | type | description |
| --- | --- | --- |
| `entry_nodes` | `list[int]` | Node indices on the entry side. |
| `exit_nodes` | `list[int]` | Node indices on the exit side. |
| `img_contours_o` | `ndarray` | RGB visualisation with contours drawn. |
| `img_preprocessed` | `ndarray` | Binary preprocessed image. |
| `cnts` | `list[ndarray]` | Simplified contours. |
| `nodes_metadata` | `pandas.DataFrame` | Per-node metadata (coords, contour index, entry/exit flags, ...). |

---

### `combra.graph.create_crack_graph`

```python
create_crack_graph(img_shape, cnts, nodes_metadata,
                   eps=100, line_eps=3, border=30, border_eps=0,
                   border_number_min=2, border_pixel=255,
                   same_node_eps=5, labels=False, labeled_line_eps=10,
                   workers=10)
```

Build the directed graph. `eps` is the maximum edge length in pixels; `line_eps` is the perpendicular tolerance used when classifying edges.

**Returns** `(g, img_contours)` — a `networkx.DiGraph` and the contour-overlay image with graph edges drawn.

**Example**

```python
from combra import graph, data

img = data.crack_images()[0][1]

(entry_nodes, exit_nodes,
 img_contours_o, img_preprocessed,
 cnts, nodes_metadata) = graph.preprocess_graph_image(
    img, border=30, disk=5, entry_ellps_w=5, exit_ellps_w=5, r=4)

g, img_contours = graph.create_crack_graph(
    img_preprocessed.shape, cnts, nodes_metadata, eps=300)
print(f'{g.number_of_nodes()} nodes, {g.number_of_edges()} edges')
```

---

### `combra.graph.get_edges`

```python
get_edges(start_node_index, nodes_metadata, process_metadata)
```

Compute outgoing edges from one node — used internally by `create_crack_graph`. You normally don't need to call it directly.

---

### `combra.graph.get_edge_type`

```python
get_edge_type(node1, node2, cnts, nodes_metadata, wc_eps=30, border_pixel=0)
```

Classify a single edge between two node indices into Co / WC-Co / WC / WC-WC.

**Returns** `int` — edge-type code (0–3).

---

### `combra.graph.get_edge_type_labeled`

```python
get_edge_type_labeled(node1, node2, nodes_metadata, line_eps=10)
```

Same as `get_edge_type` but uses hand labels carried in `nodes_metadata`. Use when you've labelled contours manually.

---

## Energies and paths

### `combra.graph.get_energies`

```python
get_energies(energy_conf, g, cnts, nodes_metadata, entry_nodes, exit_nodes,
             first_k_paths=2, parallel=False, workers=23, recalculate_paths=False)
```

Sweep an `(N, M)` grid of edge-type weights. For every grid cell, set the edge weights and run k-shortest-path between every (entry, exit) pair.

**Parameters**

| name | type | description |
| --- | --- | --- |
| `energy_conf` | `list[list[dict]]` | `(N, M)` grid where each cell is `{0: co_e, 1: wc_co_e, 2: wc_e, 3: wc_wc_e}` — edge-type weights at that grid point. |
| `g`, `cnts`, `nodes_metadata` | from `create_crack_graph` | Graph + supporting state. |
| `entry_nodes`, `exit_nodes` | `list[int]` | Endpoint pools. |
| `first_k_paths` | `int` | `2` | k for Yen's k-shortest-path. |
| `parallel` | `bool` | `False` | Use multiprocessing pool. |
| `workers` | `int` | `23` | Pool size when `parallel=True`. |
| `recalculate_paths` | `bool` | `False` | Force recompute even when a cached result exists. |

**Returns** `list[list[list[DataFrame]]]` — same shape as `energy_conf`; each cell is a list of per-pair `DataFrame`s.

**Example**

```python
import numpy as np
from combra import graph

energy_conf = np.zeros((20, 20)).tolist()
for i, en1 in enumerate(range(20)):       # Co weight
    for j, en2 in enumerate(range(20)):   # WC-Co weight
        energy_conf[i][j] = {0: en1, 1: en2, 2: 20, 3: 0}

energies_paths = graph.get_energies(
    energy_conf, g, cnts, nodes_metadata,
    entry_nodes=[0, 1, 3], exit_nodes=[63, 64, 67],
    first_k_paths=1, parallel=True, workers=20,
)
```

---

### `combra.graph.find_shortest_energy_paths`

```python
find_shortest_energy_paths(G, cnts, nodes_metadata, entry_node, exit_node, k,
                           recalculate_paths=False)
```

Find the `k` shortest paths between one entry/exit pair and return per-path lengths, energies, and edge-type breakdowns.

**Returns** `pandas.DataFrame` — one row per path with columns for total length, energy, and per-edge-type pixel fractions.

---

### `combra.graph.fixed_paths_energies`

```python
fixed_paths_energies(g, cnts, nodes_metadata, entry_nodes, exit_nodes, workers=23, ...)
```

Compute energies along a fixed set of paths (no optimisation).

---

### `combra.graph.paths_queues`

```python
paths_queues()
```

Internal queue-based path enumerator (no public arguments).

---

## Plotting

### `combra.graph.graph_plot`

```python
graph_plot(g, img_contours=None, name='graph.html', save=False,
           node_size=12, edge_width=2,
           color_dict=None, edge_width_dict=None)
```

Interactive Plotly plot of the graph overlaid on `img_contours`.

**Example**

```python
graph.graph_plot(g, img_contours=img_contours, name='crack.html', save=True)
```

---

### `combra.graph.plot_optimized_energies`

```python
plot_optimized_energies(energies_paths, path_index=0, N=5, M=5,
                        y_label='co_e', x_label='wc-co_e',
                        fontsize_h=10, fontsize_axes=50)
```

Heatmap of optimal path energies over the `(Co, WC-Co)` weight grid for path index `path_index`.

---

### `combra.graph.plot_paths`

```python
plot_paths(g, df, img_aligned, border=30)
```

Overlay the paths in `df` (output of `find_shortest_energy_paths`) on the background image.

---

### `combra.graph.plot_optimized_paths`

```python
plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10)
```

Overlay the energy-optimised paths from `get_energies` on the contour image at grid position `(param_1, param_2)`.

---

### `combra.graph.draw_tree`

```python
draw_tree(img, centres=False, leafs=False, nodes=False, bones=False)
```

Render skeleton landmarks (centres / leaves / nodes / skeleton pixels) onto a binary image.

---

## See also

- [`combra.contours`]({{< relref "/docs/contours" >}}) — the contour extractor `preprocess_graph_image` uses internally.
- [`combra.image.bresenham_line`]({{< relref "/docs/image" >}}) and friends — the geometry kernels the edge-classifier calls.
