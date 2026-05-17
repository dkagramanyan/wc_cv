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

- **image** (*ndarray*) — Source image (any channels).
- **r** (*int*, default `2`) — Marker radius for node visualisation.
- **border** (*int*, default `30`) — Padding added before extraction.
- **border_node_eps** (*int*, default `10`) — Max distance from border for a node to count as an entry/exit candidate.
- **tol** (*float*, default `5`) — Douglas–Peucker tolerance.
- **disk** (*int*, default `5`) — Median-filter footprint radius.
- **labeled_cnts** (*bool*, default `False`) — Skip binarisation; use `labels` directly.
- **labels** (*ndarray or bool*, default `False`) — Hand-labelled contour data.
- **entry_ellps_w**, **exit_ellps_w** (*int*, default `1`) — Width of the entry/exit ellipse region overlay.

**Returns**

- **entry_nodes** (*list[int]*) — Node indices on the entry side.
- **exit_nodes** (*list[int]*) — Node indices on the exit side.
- **img_contours_o** (*ndarray*) — RGB visualisation with contours drawn.
- **img_preprocessed** (*ndarray*) — Binary preprocessed image.
- **cnts** (*list[ndarray]*) — Simplified contours.
- **nodes_metadata** (*pandas.DataFrame*) — Per-node metadata (coords, contour index, entry/exit flags, …).

**Examples**

From `crack_graph/graph_unlabeled.ipynb`:

```python
from combra import graph, data

_, image = data.microstructure_images()[0]
(entry_nodes, exit_nodes,
 img_contours_o, img_preprocessed_final,
 cnts, nodes_metadata) = graph.preprocess_graph_image(
    image, border=30, disk=5, entry_ellps_w=5, exit_ellps_w=5, r=4,
)
print(f'{len(entry_nodes)} entry / {len(exit_nodes)} exit candidates')
```

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

**Parameters**

- **img_shape** (*tuple[int, int]*) — `(H, W)` of the source image.
- **cnts** (*list[ndarray]*) — Contours from `preprocess_graph_image`.
- **nodes_metadata** (*pandas.DataFrame*) — Node table from `preprocess_graph_image`.
- **eps** (*int*, default `100`) — Maximum edge length in pixels.
- **line_eps** (*int*, default `3`) — Perpendicular tolerance used when classifying edges.
- **border** (*int*, default `30`) — Image padding (must match `preprocess_graph_image`).
- **border_eps** (*int*, default `0`) — Margin from image edge that excludes detections.
- **border_number_min** (*int*, default `2`) — Minimum contour-border pixel count for an edge to count.
- **border_pixel** (*int*, default `255`) — Pixel value that marks a contour border.
- **same_node_eps** (*int*, default `5`) — Distance below which two candidate nodes are merged.
- **labels** (*ndarray or bool*, default `False`) — Hand-labelled contour data.
- **labeled_line_eps** (*int*, default `10`) — Perpendicular tolerance for the labelled-contour path.
- **workers** (*int*, default `10`) — Worker count for edge enumeration.

**Returns**

- **g** (*networkx.DiGraph*) — Built crack graph.
- **img_contours** (*ndarray*) — Contour-overlay image with graph edges drawn.

**Examples**

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

**Parameters**

- **start_node_index** (*int*) — Source node.
- **nodes_metadata** (*pandas.DataFrame*) — Node table.
- **process_metadata** (*dict*) — Internal builder state.

**Returns**

- **edges** (*list[tuple]*) — Outgoing edges with classification metadata.

**Examples**

`get_edges` is called inside `create_crack_graph`; direct invocation is only useful when implementing a custom edge enumerator:

```python
from combra import graph

# process_metadata is the internal builder state — usually obtained by patching
# create_crack_graph. For most use cases call create_crack_graph instead.
edges = graph.get_edges(start_node_index=0,
                        nodes_metadata=nodes_metadata,
                        process_metadata=process_metadata)
```

---

### `combra.graph.get_edge_type`

```python
get_edge_type(node1, node2, cnts, nodes_metadata, wc_eps=30, border_pixel=0)
```

Classify a single edge between two node indices into Co / WC-Co / WC / WC-WC.

**Parameters**

- **node1**, **node2** (*int*) — Node indices.
- **cnts** (*list[ndarray]*) — Contours.
- **nodes_metadata** (*pandas.DataFrame*) — Node table.
- **wc_eps** (*int*, default `30`) — Minimum contour-pixel count below which the edge is reclassified.
- **border_pixel** (*int*, default `0`) — Pixel value that marks a contour border.

**Returns**

- **edge_type** (*int*) — Edge-type code (0–3).

**Examples**

```python
from combra import graph

# 0=Co, 1=WC-Co, 2=WC, 3=WC-WC
edge_type = graph.get_edge_type(node1=0, node2=5,
                                cnts=cnts, nodes_metadata=nodes_metadata)
print({0: 'Co', 1: 'WC-Co', 2: 'WC', 3: 'WC-WC'}[edge_type])
```

---

### `combra.graph.get_edge_type_labeled`

```python
get_edge_type_labeled(node1, node2, nodes_metadata, line_eps=10)
```

Same as `get_edge_type` but uses hand labels carried in `nodes_metadata`. Use when you've labelled contours manually.

**Parameters**

- **node1**, **node2** (*int*) — Node indices.
- **nodes_metadata** (*pandas.DataFrame*) — Node table including labels.
- **line_eps** (*int*, default `10`) — Perpendicular tolerance.

**Returns**

- **edge_type** (*int*) — Edge-type code.

**Examples**

```python
from combra import graph

# Same code domain as get_edge_type, but reads contour-class labels from
# nodes_metadata instead of inferring from pixel values.
edge_type = graph.get_edge_type_labeled(node1=0, node2=5,
                                        nodes_metadata=nodes_metadata,
                                        line_eps=10)
```

---

## Energies and paths

### `combra.graph.get_energies`

```python
get_energies(energy_conf, g, cnts, nodes_metadata, entry_nodes, exit_nodes,
             first_k_paths=2, parallel=False, workers=23, recalculate_paths=False)
```

Sweep an `(N, M)` grid of edge-type weights. For every grid cell, set the edge weights and run k-shortest-path between every (entry, exit) pair.

**Parameters**

- **energy_conf** (*list[list[dict]]*) — `(N, M)` grid where each cell is `{0: co_e, 1: wc_co_e, 2: wc_e, 3: wc_wc_e}` — edge-type weights at that grid point.
- **g**, **cnts**, **nodes_metadata** (*from `create_crack_graph`*) — Graph + supporting state.
- **entry_nodes**, **exit_nodes** (*list[int]*) — Endpoint pools.
- **first_k_paths** (*int*, default `2`) — `k` for Yen's k-shortest-path.
- **parallel** (*bool*, default `False`) — Use multiprocessing pool.
- **workers** (*int*, default `23`) — Pool size when `parallel=True`.
- **recalculate_paths** (*bool*, default `False`) — Force recompute even when a cached result exists.

**Returns**

- **energies_paths** (*list[list[list[DataFrame]]]*) — Same shape as `energy_conf`; each cell is a list of per-pair `DataFrame`s.

**Examples**

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

**Parameters**

- **G** (*networkx.DiGraph*) — Crack graph.
- **cnts** (*list[ndarray]*) — Contours.
- **nodes_metadata** (*pandas.DataFrame*) — Node table.
- **entry_node**, **exit_node** (*int*) — Single endpoint pair.
- **k** (*int*) — Number of shortest paths to return.
- **recalculate_paths** (*bool*, default `False`) — Force recompute.

**Returns**

- **df** (*pandas.DataFrame*) — One row per path with columns for total length, energy, and per-edge-type pixel fractions.

**Examples**

```python
from combra import graph

# Set edge weights on g first (or it will use defaults), then ask for the top-3.
df = graph.find_shortest_energy_paths(
    g, cnts, nodes_metadata,
    entry_node=entry_nodes[0], exit_node=exit_nodes[0], k=3,
)
print(df[['path_len_pixel', 'energy_total']].head())
```

---

### `combra.graph.fixed_paths_energies`

```python
fixed_paths_energies(g, cnts, nodes_metadata, entry_nodes, exit_nodes, workers=23, ...)
```

Compute energies along a fixed set of paths (no optimisation).

**Examples**

```python
from combra import graph

# Use when you already have a fixed entry/exit-pair list and want their
# energies without enumerating k-shortest paths from scratch.
energies = graph.fixed_paths_energies(
    g, cnts, nodes_metadata,
    entry_nodes=[0, 1, 3], exit_nodes=[63, 64, 67], workers=8,
)
```

---

### `combra.graph.paths_queues`

```python
paths_queues()
```

Internal queue-based path enumerator (no public arguments).

**Examples**

`paths_queues` is an internal helper used by `get_energies` to multiplex (entry, exit) pairs across workers. Call `get_energies` directly — it sets up the queue for you.

---

## Plotting

### `combra.graph.graph_plot`

```python
graph_plot(g, img_contours=None, name='graph.html', save=False,
           node_size=12, edge_width=2,
           color_dict=None, edge_width_dict=None)
```

Interactive Plotly plot of the graph overlaid on `img_contours`.

**Parameters**

- **g** (*networkx.DiGraph*) — Graph to plot.
- **img_contours** (*ndarray or None*, default `None`) — Background image.
- **name** (*str*, default `'graph.html'`) — Output filename when `save=True`.
- **save** (*bool*, default `False`) — Write HTML to disk.
- **node_size** (*int*, default `12`) — Plotly marker size.
- **edge_width** (*int*, default `2`) — Default edge line width.
- **color_dict** (*dict or None*, default `None`) — `{edge_type: color}`.
- **edge_width_dict** (*dict or None*, default `None`) — `{edge_type: width}`.

**Examples**

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

**Parameters**

- **energies_paths** (*list[list[list[DataFrame]]]*) — Output of `get_energies`.
- **path_index** (*int*, default `0`) — Path rank to plot.
- **N**, **M** (*int*, default `5`) — Grid dimensions.
- **y_label**, **x_label** (*str*, default `'co_e'`, `'wc-co_e'`) — Axis labels.
- **fontsize_h**, **fontsize_axes** (*int*, default `10`, `50`) — Styling.

**Examples**

```python
from combra import graph

# energies_paths from a 20x20 (Co × WC-Co) weight grid
graph.plot_optimized_energies(
    energies_paths, path_index=0, N=20, M=20,
    y_label='co_e', x_label='wc-co_e',
)
```

---

### `combra.graph.plot_paths`

```python
plot_paths(g, df, img_aligned, border=30)
```

Overlay the paths in `df` (output of `find_shortest_energy_paths`) on the background image.

**Parameters**

- **g** (*networkx.DiGraph*) — Graph.
- **df** (*pandas.DataFrame*) — Path table.
- **img_aligned** (*ndarray*) — Background image.
- **border** (*int*, default `30`) — Padding to compensate for.

**Examples**

```python
from combra import graph

df = graph.find_shortest_energy_paths(
    g, cnts, nodes_metadata,
    entry_node=entry_nodes[0], exit_node=exit_nodes[0], k=3,
)
graph.plot_paths(g, df, img_aligned=img_contours_o, border=30)
```

---

### `combra.graph.plot_optimized_paths`

```python
plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10)
```

Overlay the energy-optimised paths from `get_energies` on the contour image at grid position `(param_1, param_2)`.

**Parameters**

- **g** (*networkx.DiGraph*) — Graph.
- **energies_paths** (*list[list[list[DataFrame]]]*) — Output of `get_energies`.
- **img_contours_o** (*ndarray*) — Background image.
- **param_1**, **param_2** (*int*, default `10`) — Grid coordinates to draw.

**Examples**

```python
from combra import graph

# Cell (10, 10) in a 20x20 (Co × WC-Co) weight grid → draw the best path there.
graph.plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10)
```

---

### `combra.graph.draw_tree`

```python
draw_tree(img, centres=False, leafs=False, nodes=False, bones=False)
```

Render skeleton landmarks (centres / leaves / nodes / skeleton pixels) onto a binary image.

**Parameters**

- **img** (*ndarray*) — Binary input.
- **centres**, **leafs**, **nodes**, **bones** (*bool*, default `False`) — Draw each landmark type.

**Returns**

- **rendered** (*ndarray*) — Annotated image.

**Examples**

```python
from combra import graph, image, data

_, img = data.microstructure_images()[0]
binary = image.do_otsu(img)
annotated = graph.draw_tree(binary, centres=True, nodes=True, bones=True)
```

---

## See also

- [`combra.contours`]({{< relref "/docs/contours" >}}) — the contour extractor `preprocess_graph_image` uses internally.
- [`combra.image.bresenham_line`]({{< relref "/docs/image" >}}) and friends — the geometry kernels the edge-classifier calls.
