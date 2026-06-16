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
combra.graph.preprocess_graph_image(image, r=2, border=30, border_node_eps=10,
                                    tol=5, disk=5, labeled_cnts=False, labels=False,
                                    entry_ellps_w=1, exit_ellps_w=1)
                                    → tuple[list[int], list[int], ndarray, ndarray, list[ndarray], dict]
```

Median-filter, Otsu, and contour-extract the input. Pass `labeled_cnts=True` (and `labels`) when you have hand-annotated contours and want to skip binarisation.

**Parameters**

- **image** (*ndarray*) – Source image (any channels).
- **r** (*int, optional*) – Marker radius for node visualisation. Default: `2`.
- **border** (*int, optional*) – Padding added before extraction. Default: `30`.
- **border_node_eps** (*int, optional*) – Max distance from border for a node to count as an entry/exit candidate. Default: `10`.
- **tol** (*float, optional*) – Douglas–Peucker tolerance. Default: `5`.
- **disk** (*int, optional*) – Median-filter footprint radius. Default: `5`.
- **labeled_cnts** (*bool, optional*) – Skip binarisation; use `labels` directly. Default: `False`.
- **labels** (*ndarray or bool, optional*) – Hand-labelled contour data. Default: `False`.
- **entry_ellps_w** (*int, optional*) – Width of the entry ellipse region overlay. Default: `1`.
- **exit_ellps_w** (*int, optional*) – Width of the exit ellipse region overlay. Default: `1`.

**Returns**

- **entry_nodes** (*list[int]*) – Node indices on the entry side.
- **exit_nodes** (*list[int]*) – Node indices on the exit side.
- **img_preprocessed** (*ndarray*) – Binary preprocessed image (a copy used as the edge-drawing canvas). Returned third.
- **img_contours_o** (*ndarray*) – RGB visualisation with contours and entry/exit nodes drawn. Returned fourth.
- **cnts** (*list[ndarray]*) – Simplified contours.
- **nodes_metadata** (*dict*) – Per-node lookup tables (global node coords, global/local contour indices, coord→index maps, and `labels`/`contour_index2label` when labelled).

**Return type**

*tuple(list[int], list[int], ndarray, ndarray, list[ndarray], dict)*

**Example**

From `crack_graph/graph_unlabeled.ipynb`:

```python
>>> from combra import graph, data
>>> _, image = data.microstructure_images()[0]
>>> (entry_nodes, exit_nodes,
...  img_preprocessed, img_contours_o,
...  cnts, nodes_metadata) = graph.preprocess_graph_image(
...     image, border=30, disk=5, entry_ellps_w=5, exit_ellps_w=5, r=4,
... )
>>> print(f'{len(entry_nodes)} entry / {len(exit_nodes)} exit candidates')
```

---

### `combra.graph.create_crack_graph`

```python
combra.graph.create_crack_graph(img_shape, cnts, nodes_metadata,
                                eps=100, line_eps=3, border=30, border_eps=0,
                                border_number_min=2, border_pixel=255,
                                same_node_eps=5, labels=False, labeled_line_eps=10,
                                workers=10) → tuple[networkx.DiGraph, ndarray]
```

Build the directed graph. `eps` is the maximum edge length in pixels; `line_eps` is the perpendicular tolerance used when classifying edges.

**Parameters**

- **img_shape** (*tuple[int, int]*) – `(H, W)` of the source image.
- **cnts** (*list[ndarray]*) – Contours from `preprocess_graph_image`.
- **nodes_metadata** (*dict*) – Node lookup tables from `preprocess_graph_image`.
- **eps** (*int, optional*) – Maximum edge length in pixels. Default: `100`.
- **line_eps** (*int, optional*) – Perpendicular tolerance used when classifying edges. Default: `3`.
- **border** (*int, optional*) – Image padding (must match `preprocess_graph_image`). Default: `30`.
- **border_eps** (*int, optional*) – Margin from image edge that excludes detections. Default: `0`.
- **border_number_min** (*int, optional*) – Minimum contour-border pixel count for an edge to count. Default: `2`.
- **border_pixel** (*int, optional*) – Pixel value that marks a contour border. Default: `255`.
- **same_node_eps** (*int, optional*) – Distance below which two candidate nodes are merged. Default: `5`.
- **labels** (*ndarray or bool, optional*) – Hand-labelled contour data. Default: `False`.
- **labeled_line_eps** (*int, optional*) – Perpendicular tolerance for the labelled-contour path. Default: `10`.
- **workers** (*int, optional*) – Worker count for edge enumeration. Default: `10`.

**Returns**

- **g** (*networkx.DiGraph*) – Built crack graph.
- **img_contours** (*ndarray*) – Contour-overlay image with graph edges drawn.

**Return type**

*tuple(networkx.DiGraph, ndarray)*

**Example**

```python
>>> from combra import graph, data
>>> img = data.crack_images()[0][1]
>>> (entry_nodes, exit_nodes,
...  img_preprocessed, img_contours_o,
...  cnts, nodes_metadata) = graph.preprocess_graph_image(
...     img, border=30, disk=5, entry_ellps_w=5, exit_ellps_w=5, r=4)
>>> g, img_contours = graph.create_crack_graph(
...     img_preprocessed.shape, cnts, nodes_metadata, eps=300)
>>> print(f'{g.number_of_nodes()} nodes, {g.number_of_edges()} edges')
```

---

### `combra.graph.get_edges`

```python
combra.graph.get_edges(start_node_index, nodes_metadata, process_metadata) → list[dict]
```

Compute outgoing edges from one node — used internally by `create_crack_graph`. You normally don't need to call it directly.

**Parameters**

- **start_node_index** (*int*) – Source node.
- **nodes_metadata** (*dict*) – Node lookup tables.
- **process_metadata** (*dict*) – Internal builder state.

**Returns**

- **edges** (*list[dict]*) – Outgoing edges with classification metadata.

**Return type**

*list[dict]*

**Example**

`get_edges` is called inside `create_crack_graph`; direct invocation is only useful when implementing a custom edge enumerator:

```python
>>> from combra import graph
>>> # process_metadata is the internal builder state — usually obtained by patching
>>> # create_crack_graph. For most use cases call create_crack_graph instead.
>>> edges = graph.get_edges(start_node_index=0,
...                         nodes_metadata=nodes_metadata,
...                         process_metadata=process_metadata)
```

---

### `combra.graph.get_edge_type`

```python
combra.graph.get_edge_type(node1, node2, cnts, nodes_metadata, wc_eps=30,
                           border_pixel=0) → int
```

Classify a single edge between two node indices into Co / WC-Co / WC / WC-WC.

**Parameters**

- **node1** (*int*) – First node index.
- **node2** (*int*) – Second node index.
- **cnts** (*list[ndarray]*) – Contours.
- **nodes_metadata** (*dict*) – Node lookup tables.
- **wc_eps** (*int, optional*) – Minimum contour-pixel count below which the edge is reclassified. Default: `30`.
- **border_pixel** (*int, optional*) – Pixel value that marks a contour border. Default: `0`.

**Returns**

- **edge_type** (*int*) – Edge-type code (0–3).

**Return type**

*int*

**Example**

```python
>>> from combra import graph
>>> # 0=Co, 1=WC-Co, 2=WC, 3=WC-WC
>>> edge_type = graph.get_edge_type(node1=0, node2=5,
...                                 cnts=cnts, nodes_metadata=nodes_metadata)
>>> print({0: 'Co', 1: 'WC-Co', 2: 'WC', 3: 'WC-WC'}[edge_type])
```

---

### `combra.graph.get_edge_type_labeled`

```python
combra.graph.get_edge_type_labeled(node1, node2, nodes_metadata, line_eps=10) → int
```

Same as `get_edge_type` but uses hand labels carried in `nodes_metadata`. Use when you've labelled contours manually.

**Parameters**

- **node1** (*int*) – First node index.
- **node2** (*int*) – Second node index.
- **nodes_metadata** (*dict*) – Node lookup tables including `labels`.
- **line_eps** (*int, optional*) – Perpendicular tolerance. Default: `10`.

**Returns**

- **edge_type** (*int*) – Edge-type code.

**Return type**

*int*

**Example**

```python
>>> from combra import graph
>>> # Same code domain as get_edge_type, but reads contour-class labels from
>>> # nodes_metadata instead of inferring from pixel values.
>>> edge_type = graph.get_edge_type_labeled(node1=0, node2=5,
...                                         nodes_metadata=nodes_metadata,
...                                         line_eps=10)
```

---

## Energies and paths

### `combra.graph.get_energies`

```python
combra.graph.get_energies(energy_conf, g, cnts, nodes_metadata, entry_nodes, exit_nodes,
                          first_k_paths=2, parallel=False, workers=23,
                          recalculate_paths=False) → list[list[list[DataFrame]]]
```

Sweep an `(N, M)` grid of edge-type weights. For every grid cell, set the edge weights and run k-shortest-path between every (entry, exit) pair.

**Parameters**

- **energy_conf** (*list[list[dict]]*) – `(N, M)` grid where each cell is `{0: co_e, 1: wc_co_e, 2: wc_e, 3: wc_wc_e}` — edge-type weights at that grid point.
- **g** (*networkx.DiGraph*) – Graph from `create_crack_graph`.
- **cnts** (*list[ndarray]*) – Contours from `create_crack_graph`.
- **nodes_metadata** (*dict*) – Node lookup tables from `create_crack_graph`.
- **entry_nodes** (*list[int]*) – Entry endpoint pool.
- **exit_nodes** (*list[int]*) – Exit endpoint pool.
- **first_k_paths** (*int, optional*) – `k` for Yen's k-shortest-path. Default: `2`.
- **parallel** (*bool, optional*) – Use multiprocessing pool. Default: `False`.
- **workers** (*int, optional*) – Pool size when `parallel=True`. Default: `23`.
- **recalculate_paths** (*bool, optional*) – Force recompute even when a cached result exists. Default: `False`.

**Returns**

- **energies_paths** (*list[list[list[DataFrame]]]*) – Same shape as `energy_conf`; each cell is a list of per-pair `DataFrame`s.

**Return type**

*list[list[list[DataFrame]]]*

**Example**

```python
>>> import numpy as np
>>> from combra import graph
>>> energy_conf = np.zeros((20, 20)).tolist()
>>> for i, en1 in enumerate(range(20)):       # Co weight
...     for j, en2 in enumerate(range(20)):   # WC-Co weight
...         energy_conf[i][j] = {0: en1, 1: en2, 2: 20, 3: 0}
>>> energies_paths = graph.get_energies(
...     energy_conf, g, cnts, nodes_metadata,
...     entry_nodes=[0, 1, 3], exit_nodes=[63, 64, 67],
...     first_k_paths=1, parallel=True, workers=20,
... )
```

---

### `combra.graph.find_shortest_energy_paths`

```python
combra.graph.find_shortest_energy_paths(G, cnts, nodes_metadata, entry_node, exit_node, k,
                                        recalculate_paths=False) → pandas.DataFrame
```

Find the `k` shortest paths between one entry/exit pair and return per-path lengths, energies, and edge-type breakdowns.

**Parameters**

- **G** (*networkx.DiGraph*) – Crack graph.
- **cnts** (*list[ndarray]*) – Contours.
- **nodes_metadata** (*dict*) – Node lookup tables.
- **entry_node** (*int*) – Entry endpoint.
- **exit_node** (*int*) – Exit endpoint.
- **k** (*int*) – Number of shortest paths to return.
- **recalculate_paths** (*bool, optional*) – Force recompute. Default: `False`.

**Returns**

- **df** (*pandas.DataFrame*) – One row per path with columns for total length, energy, and per-edge-type pixel fractions.

**Return type**

*pandas.DataFrame*

**Example**

```python
>>> from combra import graph
>>> # Set edge weights on g first (or it will use defaults), then ask for the top-3.
>>> df = graph.find_shortest_energy_paths(
...     g, cnts, nodes_metadata,
...     entry_node=entry_nodes[0], exit_node=exit_nodes[0], k=3,
... )
>>> print(df[['path_len_pixel', 'energy_total']].head())
```

---

### `combra.graph.fixed_paths_energies`

```python
combra.graph.fixed_paths_energies(g, cnts, nodes_metadata, entry_nodes, exit_nodes,
                                  workers=23, ...) → list[list[list[DataFrame]]]
```

Compute energies along a fixed set of paths (no optimisation).

**Returns**

- **energies_paths_recalculated** (*list[list[list[DataFrame]]]*) – Per-grid-cell lists of fixed-path `DataFrame`s.

**Return type**

*list[list[list[DataFrame]]]*

**Example**

```python
>>> from combra import graph
>>> # Use when you already have a fixed entry/exit-pair list and want their
>>> # energies without enumerating k-shortest paths from scratch.
>>> energies = graph.fixed_paths_energies(
...     g, cnts, nodes_metadata,
...     entry_nodes=[0, 1, 3], exit_nodes=[63, 64, 67], workers=8,
... )
```

---

### `combra.graph.paths_queues`

```python
combra.graph.paths_queues(g, entry_nodes, exit_nodes, workers=23) → list
```

Internal queue-based path enumerator — enumerates all simple paths for every `(entry, exit)` pair across a worker pool.

**Parameters**

- **g** (*networkx.DiGraph*) – Crack graph.
- **entry_nodes** (*list[int]*) – Entry endpoint pool to pair up.
- **exit_nodes** (*list[int]*) – Exit endpoint pool to pair up.
- **workers** (*int, optional*) – Pool size. Default: `23`.

**Returns**

- **results** (*list*) – Flattened per-pair path enumeration results.

**Return type**

*list*

**Example**

`paths_queues` is an internal helper used by `get_energies` to multiplex (entry, exit) pairs across workers. Call `get_energies` directly — it sets up the queue for you.

---

## Plotting

### `combra.graph.graph_plot`

```python
combra.graph.graph_plot(g, img_contours=None, name='graph.html', save=False,
                        node_size=12, edge_width=2,
                        color_dict=None, edge_width_dict=None)
                        → plotly.graph_objects.Figure
```

Interactive Plotly plot of the graph overlaid on `img_contours`.

**Parameters**

- **g** (*networkx.DiGraph*) – Graph to plot.
- **img_contours** (*ndarray or None, optional*) – Background image. Default: `None`.
- **name** (*str, optional*) – Output filename when `save=True`. Default: `'graph.html'`.
- **save** (*bool, optional*) – Write HTML to disk. Default: `False`.
- **node_size** (*int, optional*) – Plotly marker size. Default: `12`.
- **edge_width** (*int, optional*) – Default edge line width. Default: `2`.
- **color_dict** (*dict or None, optional*) – `{edge_type: color}`. Default: `None`.
- **edge_width_dict** (*dict or None, optional*) – `{edge_type: width}`. Default: `None`.

**Returns**

- **fig** (*plotly.graph_objects.Figure*) – The assembled graph figure.

**Return type**

*plotly.graph_objects.Figure*

**Example**

```python
>>> graph.graph_plot(g, img_contours=img_contours, name='crack.html', save=True)
```

---

### `combra.graph.plot_optimized_energies`

```python
combra.graph.plot_optimized_energies(energies, path_index=0, N=6, M=6,
                                     name='test.jpg', y_label='co_co_e', x_label='wc_co_e',
                                     save=False, fixed_paths=False,
                                     fontsize_h=10, fontsize_axes=30) → None
```

Heatmap of optimal path energies over the `(Co, WC-Co)` weight grid for path index `path_index`.

**Parameters**

- **energies** (*list[list[list[DataFrame]]]*) – Output of `get_energies`.
- **path_index** (*int, optional*) – Path rank to plot. Default: `0`.
- **N** (*int, optional*) – Grid rows. Default: `6`.
- **M** (*int, optional*) – Grid columns. Default: `6`.
- **name** (*str, optional*) – Output filename when `save=True`. Default: `'test.jpg'`.
- **y_label** (*str, optional*) – Y-axis label. Default: `'co_co_e'`.
- **x_label** (*str, optional*) – X-axis label. Default: `'wc_co_e'`.
- **save** (*bool, optional*) – Write the figure to `name`. Default: `False`.
- **fixed_paths** (*bool, optional*) – Read cells as fixed-path results (output of `fixed_paths_energies`) instead of optimised k-shortest paths. Default: `False`.
- **fontsize_h** (*int, optional*) – Heatmap annotation font size. Default: `10`.
- **fontsize_axes** (*int, optional*) – Axis label font size. Default: `30`.

**Returns**

None. Renders the heatmap and, when `save=True`, writes it to `name`.

**Return type**

*None*

**Example**

```python
>>> from combra import graph
>>> # energies_paths from a 20x20 (Co × WC-Co) weight grid
>>> graph.plot_optimized_energies(
...     energies_paths, path_index=0, N=20, M=20,
...     y_label='co_co_e', x_label='wc_co_e',
... )
```

---

### `combra.graph.plot_paths`

```python
combra.graph.plot_paths(g, df, img_aligned, border=30) → None
```

Overlay the paths in `df` (output of `find_shortest_energy_paths`) on the background image.

**Parameters**

- **g** (*networkx.DiGraph*) – Graph.
- **df** (*pandas.DataFrame*) – Path table.
- **img_aligned** (*ndarray*) – Background image.
- **border** (*int, optional*) – Padding to compensate for. Default: `30`.

**Returns**

None. Renders the path overlay.

**Return type**

*None*

**Example**

```python
>>> from combra import graph
>>> df = graph.find_shortest_energy_paths(
...     g, cnts, nodes_metadata,
...     entry_node=entry_nodes[0], exit_node=exit_nodes[0], k=3,
... )
>>> graph.plot_paths(g, df, img_aligned=img_contours_o, border=30)
```

---

### `combra.graph.plot_optimized_paths`

```python
combra.graph.plot_optimized_paths(g, energies_paths, img_contours_o,
                                  param_1=10, param_2=10) → None
```

Overlay the energy-optimised paths from `get_energies` on the contour image at grid position `(param_1, param_2)`.

**Parameters**

- **g** (*networkx.DiGraph*) – Graph.
- **energies_paths** (*list[list[list[DataFrame]]]*) – Output of `get_energies`.
- **img_contours_o** (*ndarray*) – Background image.
- **param_1** (*int, optional*) – Grid row coordinate to draw. Default: `10`.
- **param_2** (*int, optional*) – Grid column coordinate to draw. Default: `10`.

**Returns**

None. Renders the optimised-path overlay.

**Return type**

*None*

**Example**

```python
>>> from combra import graph
>>> # Cell (10, 10) in a 20x20 (Co × WC-Co) weight grid → draw the best path there.
>>> graph.plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10)
```

---

### `combra.graph.draw_tree`

```python
combra.graph.draw_tree(img, centres=False, leafs=False, nodes=False, bones=False) → ndarray
```

Render skeleton landmarks (centres / leaves / nodes / skeleton pixels) onto a binary image.

**Parameters**

- **img** (*ndarray*) – Binary input.
- **centres** (*bool, optional*) – Draw centre landmarks. Default: `False`.
- **leafs** (*bool, optional*) – Draw leaf landmarks. Default: `False`.
- **nodes** (*bool, optional*) – Draw node landmarks. Default: `False`.
- **bones** (*bool, optional*) – Draw skeleton pixels. Default: `False`.

**Returns**

- **rendered** (*ndarray*) – Annotated image.

**Return type**

*ndarray*

**Example**

```python
>>> from combra import graph, image, data
>>> _, img = data.microstructure_images()[0]
>>> binary = image.do_otsu(img)
>>> annotated = graph.draw_tree(binary, centres=True, nodes=True, bones=True)
```

---

## See also

- [`combra.contours`]({{< relref "/docs/contours" >}}) — the contour extractor `preprocess_graph_image` uses internally.
- [`combra.image.bresenham_line`]({{< relref "/docs/image" >}}) and friends — the geometry kernels the edge-classifier calls.
