# combra.graph

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

````{py:function} combra.graph.preprocess_graph_image(image, r=2, border=30, border_node_eps=10, tol=5, disk=5, labeled_cnts=False, labels=False, entry_ellps_w=1, exit_ellps_w=1) -> tuple[list[int], list[int], ndarray, ndarray, list[ndarray], dict]

Median-filter, Otsu, and contour-extract the input. Pass ``labeled_cnts=True`` (and ``labels``) when you have hand-annotated contours and want to skip binarisation.

:param image: Source image (any channels).
:type image: ndarray
:param r: Marker radius for node visualisation. Default: ``2``.
:type r: int, optional
:param border: Padding added before extraction. Default: ``30``.
:type border: int, optional
:param border_node_eps: Max distance from border for a node to count as an entry/exit candidate. Default: ``10``.
:type border_node_eps: int, optional
:param tol: Douglas–Peucker tolerance. Default: ``5``.
:type tol: float, optional
:param disk: Median-filter footprint radius. Default: ``5``.
:type disk: int, optional
:param labeled_cnts: Skip binarisation; use ``labels`` directly. Default: ``False``.
:type labeled_cnts: bool, optional
:param labels: Hand-labelled contour data. Default: ``False``.
:type labels: ndarray or bool, optional
:param entry_ellps_w: Width of the entry ellipse region overlay. Default: ``1``.
:type entry_ellps_w: int, optional
:param exit_ellps_w: Width of the exit ellipse region overlay. Default: ``1``.
:type exit_ellps_w: int, optional
:returns: **entry_nodes** (*list[int]*) – Node indices on the entry side; and **exit_nodes** (*list[int]*) – Node indices on the exit side; and **img_preprocessed** (*ndarray*) – Binary preprocessed image (a copy used as the edge-drawing canvas). Returned third; and **img_contours_o** (*ndarray*) – RGB visualisation with contours and entry/exit nodes drawn. Returned fourth; and **cnts** (*list[ndarray]*) – Simplified contours; and **nodes_metadata** (*dict*) – Per-node lookup tables (global node coords, global/local contour indices, coord→index maps, and ``labels``/``contour_index2label`` when labelled).
:rtype: tuple(list[int], list[int], ndarray, ndarray, list[ndarray], dict)

**Example**

From ``crack_graph/graph_unlabeled.ipynb``:

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
````

````{py:function} combra.graph.create_crack_graph(img_shape, cnts, nodes_metadata, eps=100, line_eps=3, border=30, border_eps=0, border_number_min=2, border_pixel=255, same_node_eps=5, labels=False, labeled_line_eps=10, workers=10) -> tuple[networkx.DiGraph, ndarray]

Build the directed graph. ``eps`` is the maximum edge length in pixels; ``line_eps`` is the perpendicular tolerance used when classifying edges.

:param img_shape: ``(H, W)`` of the source image.
:type img_shape: tuple[int, int]
:param cnts: Contours from ``preprocess_graph_image``.
:type cnts: list[ndarray]
:param nodes_metadata: Node lookup tables from ``preprocess_graph_image``.
:type nodes_metadata: dict
:param eps: Maximum edge length in pixels. Default: ``100``.
:type eps: int, optional
:param line_eps: Perpendicular tolerance used when classifying edges. Default: ``3``.
:type line_eps: int, optional
:param border: Image padding (must match ``preprocess_graph_image``). Default: ``30``.
:type border: int, optional
:param border_eps: Margin from image edge that excludes detections. Default: ``0``.
:type border_eps: int, optional
:param border_number_min: Minimum contour-border pixel count for an edge to count. Default: ``2``.
:type border_number_min: int, optional
:param border_pixel: Pixel value that marks a contour border. Default: ``255``.
:type border_pixel: int, optional
:param same_node_eps: Distance below which two candidate nodes are merged. Default: ``5``.
:type same_node_eps: int, optional
:param labels: Hand-labelled contour data. Default: ``False``.
:type labels: ndarray or bool, optional
:param labeled_line_eps: Perpendicular tolerance for the labelled-contour path. Default: ``10``.
:type labeled_line_eps: int, optional
:param workers: Worker count for edge enumeration. Default: ``10``.
:type workers: int, optional
:returns: **g** (*networkx.DiGraph*) – Built crack graph; and **img_contours** (*ndarray*) – Contour-overlay image with graph edges drawn.
:rtype: tuple(networkx.DiGraph, ndarray)

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
````

````{py:function} combra.graph.get_edges(start_node_index, nodes_metadata, process_metadata) -> list[dict]

Compute outgoing edges from one node — used internally by ``create_crack_graph``. You normally don't need to call it directly.

:param start_node_index: Source node.
:type start_node_index: int
:param nodes_metadata: Node lookup tables.
:type nodes_metadata: dict
:param process_metadata: Internal builder state.
:type process_metadata: dict
:returns: **edges** (*list[dict]*) – Outgoing edges with classification metadata.
:rtype: list[dict]

**Example**

``get_edges`` is called inside ``create_crack_graph``; direct invocation is only useful when implementing a custom edge enumerator:

```python
>>> from combra import graph
>>> # process_metadata is the internal builder state — usually obtained by patching
>>> # create_crack_graph. For most use cases call create_crack_graph instead.
>>> edges = graph.get_edges(start_node_index=0,
...                         nodes_metadata=nodes_metadata,
...                         process_metadata=process_metadata)
```
````

````{py:function} combra.graph.get_edge_type(node1, node2, cnts, nodes_metadata, wc_eps=30, border_pixel=0) -> int

Classify a single edge between two node indices into Co / WC-Co / WC / WC-WC.

:param node1: First node index.
:type node1: int
:param node2: Second node index.
:type node2: int
:param cnts: Contours.
:type cnts: list[ndarray]
:param nodes_metadata: Node lookup tables.
:type nodes_metadata: dict
:param wc_eps: Minimum contour-pixel count below which the edge is reclassified. Default: ``30``.
:type wc_eps: int, optional
:param border_pixel: Pixel value that marks a contour border. Default: ``0``.
:type border_pixel: int, optional
:returns: **edge_type** (*int*) – Edge-type code (0–3).
:rtype: int

**Example**

```python
>>> from combra import graph
>>> # 0=Co, 1=WC-Co, 2=WC, 3=WC-WC
>>> edge_type = graph.get_edge_type(node1=0, node2=5,
...                                 cnts=cnts, nodes_metadata=nodes_metadata)
>>> print({0: 'Co', 1: 'WC-Co', 2: 'WC', 3: 'WC-WC'}[edge_type])
```
````

````{py:function} combra.graph.get_edge_type_labeled(node1, node2, nodes_metadata, line_eps=10) -> int

Same as ``get_edge_type`` but uses hand labels carried in ``nodes_metadata``. Use when you've labelled contours manually.

:param node1: First node index.
:type node1: int
:param node2: Second node index.
:type node2: int
:param nodes_metadata: Node lookup tables including ``labels``.
:type nodes_metadata: dict
:param line_eps: Perpendicular tolerance. Default: ``10``.
:type line_eps: int, optional
:returns: **edge_type** (*int*) – Edge-type code.
:rtype: int

**Example**

```python
>>> from combra import graph
>>> # Same code domain as get_edge_type, but reads contour-class labels from
>>> # nodes_metadata instead of inferring from pixel values.
>>> edge_type = graph.get_edge_type_labeled(node1=0, node2=5,
...                                         nodes_metadata=nodes_metadata,
...                                         line_eps=10)
```
````

## Energies and paths

````{py:function} combra.graph.get_energies(energy_conf, g, cnts, nodes_metadata, entry_nodes, exit_nodes, first_k_paths=2, parallel=False, workers=23, recalculate_paths=False) -> list[list[list[DataFrame]]]

Sweep an ``(N, M)`` grid of edge-type weights. For every grid cell, set the edge weights and run k-shortest-path between every (entry, exit) pair.

:param energy_conf: ``(N, M)`` grid where each cell is ``{0: co_e, 1: wc_co_e, 2: wc_e, 3: wc_wc_e}`` — edge-type weights at that grid point.
:type energy_conf: list[list[dict]]
:param g: Graph from ``create_crack_graph``.
:type g: networkx.DiGraph
:param cnts: Contours from ``create_crack_graph``.
:type cnts: list[ndarray]
:param nodes_metadata: Node lookup tables from ``create_crack_graph``.
:type nodes_metadata: dict
:param entry_nodes: Entry endpoint pool.
:type entry_nodes: list[int]
:param exit_nodes: Exit endpoint pool.
:type exit_nodes: list[int]
:param first_k_paths: ``k`` for Yen's k-shortest-path. Default: ``2``.
:type first_k_paths: int, optional
:param parallel: Use multiprocessing pool. Default: ``False``.
:type parallel: bool, optional
:param workers: Pool size when ``parallel=True``. Default: ``23``.
:type workers: int, optional
:param recalculate_paths: Force recompute even when a cached result exists. Default: ``False``.
:type recalculate_paths: bool, optional
:returns: **energies_paths** (*list[list[list[DataFrame]]]*) – Same shape as ``energy_conf``; each cell is a list of per-pair ``DataFrame``s.
:rtype: list[list[list[DataFrame]]]

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
````

````{py:function} combra.graph.find_shortest_energy_paths(G, cnts, nodes_metadata, entry_node, exit_node, k, recalculate_paths=False) -> pandas.DataFrame

Find the ``k`` shortest paths between one entry/exit pair and return per-path lengths, energies, and edge-type breakdowns.

:param G: Crack graph.
:type G: networkx.DiGraph
:param cnts: Contours.
:type cnts: list[ndarray]
:param nodes_metadata: Node lookup tables.
:type nodes_metadata: dict
:param entry_node: Entry endpoint.
:type entry_node: int
:param exit_node: Exit endpoint.
:type exit_node: int
:param k: Number of shortest paths to return.
:type k: int
:param recalculate_paths: Force recompute. Default: ``False``.
:type recalculate_paths: bool, optional
:returns: **df** (*pandas.DataFrame*) – One row per path with columns for total length, energy, and per-edge-type pixel fractions.
:rtype: pandas.DataFrame

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
````

````{py:function} combra.graph.fixed_paths_energies(g, cnts, nodes_metadata, entry_nodes, exit_nodes, workers=23, ...) -> list[list[list[DataFrame]]]

Compute energies along a fixed set of paths (no optimisation).

:returns: **energies_paths_recalculated** (*list[list[list[DataFrame]]]*) – Per-grid-cell lists of fixed-path ``DataFrame``s.
:rtype: list[list[list[DataFrame]]]

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
````

````{py:function} combra.graph.paths_queues(g, entry_nodes, exit_nodes, workers=23) -> list

Internal queue-based path enumerator — enumerates all simple paths for every ``(entry, exit)`` pair across a worker pool.

:param g: Crack graph.
:type g: networkx.DiGraph
:param entry_nodes: Entry endpoint pool to pair up.
:type entry_nodes: list[int]
:param exit_nodes: Exit endpoint pool to pair up.
:type exit_nodes: list[int]
:param workers: Pool size. Default: ``23``.
:type workers: int, optional
:returns: **results** (*list*) – Flattened per-pair path enumeration results.
:rtype: list

**Example**

``paths_queues`` is an internal helper used by ``get_energies`` to multiplex (entry, exit) pairs across workers. Call ``get_energies`` directly — it sets up the queue for you.
````

## Plotting

````{py:function} combra.graph.graph_plot(g, img_contours=None, name='graph.html', save=False, node_size=12, edge_width=2, color_dict=None, edge_width_dict=None) -> plotly.graph_objects.Figure

Interactive Plotly plot of the graph overlaid on ``img_contours``.

:param g: Graph to plot.
:type g: networkx.DiGraph
:param img_contours: Background image. Default: ``None``.
:type img_contours: ndarray or None, optional
:param name: Output filename when ``save=True``. Default: ``'graph.html'``.
:type name: str, optional
:param save: Write HTML to disk. Default: ``False``.
:type save: bool, optional
:param node_size: Plotly marker size. Default: ``12``.
:type node_size: int, optional
:param edge_width: Default edge line width. Default: ``2``.
:type edge_width: int, optional
:param color_dict: ``{edge_type: color}``. Default: ``None``.
:type color_dict: dict or None, optional
:param edge_width_dict: ``{edge_type: width}``. Default: ``None``.
:type edge_width_dict: dict or None, optional
:returns: **fig** (*plotly.graph_objects.Figure*) – The assembled graph figure.
:rtype: plotly.graph_objects.Figure

**Example**

```python
>>> graph.graph_plot(g, img_contours=img_contours, name='crack.html', save=True)
```
````

````{py:function} combra.graph.plot_optimized_energies(energies, path_index=0, N=6, M=6, name='test.jpg', y_label='co_co_e', x_label='wc_co_e', save=False, fixed_paths=False, fontsize_h=10, fontsize_axes=30) -> None

Heatmap of optimal path energies over the ``(Co, WC-Co)`` weight grid for path index ``path_index``.

:param energies: Output of ``get_energies``.
:type energies: list[list[list[DataFrame]]]
:param path_index: Path rank to plot. Default: ``0``.
:type path_index: int, optional
:param N: Grid rows. Default: ``6``.
:type N: int, optional
:param M: Grid columns. Default: ``6``.
:type M: int, optional
:param name: Output filename when ``save=True``. Default: ``'test.jpg'``.
:type name: str, optional
:param y_label: Y-axis label. Default: ``'co_co_e'``.
:type y_label: str, optional
:param x_label: X-axis label. Default: ``'wc_co_e'``.
:type x_label: str, optional
:param save: Write the figure to ``name``. Default: ``False``.
:type save: bool, optional
:param fixed_paths: Read cells as fixed-path results (output of ``fixed_paths_energies``) instead of optimised k-shortest paths. Default: ``False``.
:type fixed_paths: bool, optional
:param fontsize_h: Heatmap annotation font size. Default: ``10``.
:type fontsize_h: int, optional
:param fontsize_axes: Axis label font size. Default: ``30``.
:type fontsize_axes: int, optional
:returns: Nothing. Renders the heatmap and, when ``save=True``, writes it to ``name``.
:rtype: None

**Example**

```python
>>> from combra import graph
>>> # energies_paths from a 20x20 (Co × WC-Co) weight grid
>>> graph.plot_optimized_energies(
...     energies_paths, path_index=0, N=20, M=20,
...     y_label='co_co_e', x_label='wc_co_e',
... )
```
````

````{py:function} combra.graph.plot_paths(g, df, img_aligned, border=30) -> None

Overlay the paths in ``df`` (output of ``find_shortest_energy_paths``) on the background image.

:param g: Graph.
:type g: networkx.DiGraph
:param df: Path table.
:type df: pandas.DataFrame
:param img_aligned: Background image.
:type img_aligned: ndarray
:param border: Padding to compensate for. Default: ``30``.
:type border: int, optional
:returns: Nothing. Renders the path overlay.
:rtype: None

**Example**

```python
>>> from combra import graph
>>> df = graph.find_shortest_energy_paths(
...     g, cnts, nodes_metadata,
...     entry_node=entry_nodes[0], exit_node=exit_nodes[0], k=3,
... )
>>> graph.plot_paths(g, df, img_aligned=img_contours_o, border=30)
```
````

````{py:function} combra.graph.plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10) -> None

Overlay the energy-optimised paths from ``get_energies`` on the contour image at grid position ``(param_1, param_2)``.

:param g: Graph.
:type g: networkx.DiGraph
:param energies_paths: Output of ``get_energies``.
:type energies_paths: list[list[list[DataFrame]]]
:param img_contours_o: Background image.
:type img_contours_o: ndarray
:param param_1: Grid row coordinate to draw. Default: ``10``.
:type param_1: int, optional
:param param_2: Grid column coordinate to draw. Default: ``10``.
:type param_2: int, optional
:returns: Nothing. Renders the optimised-path overlay.
:rtype: None

**Example**

```python
>>> from combra import graph
>>> # Cell (10, 10) in a 20x20 (Co × WC-Co) weight grid → draw the best path there.
>>> graph.plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10)
```
````

````{py:function} combra.graph.draw_tree(img, centres=False, leafs=False, nodes=False, bones=False) -> ndarray

Render skeleton landmarks (centres / leaves / nodes / skeleton pixels) onto a binary image.

:param img: Binary input.
:type img: ndarray
:param centres: Draw centre landmarks. Default: ``False``.
:type centres: bool, optional
:param leafs: Draw leaf landmarks. Default: ``False``.
:type leafs: bool, optional
:param nodes: Draw node landmarks. Default: ``False``.
:type nodes: bool, optional
:param bones: Draw skeleton pixels. Default: ``False``.
:type bones: bool, optional
:returns: **rendered** (*ndarray*) – Annotated image.
:rtype: ndarray

**Example**

```python
>>> from combra import graph, image, data
>>> _, img = data.microstructure_images()[0]
>>> binary = image.do_otsu(img)
>>> annotated = graph.draw_tree(binary, centres=True, nodes=True, bones=True)
```
````

## See also

- {doc}`combra.contours <contours>` — the contour extractor `preprocess_graph_image` uses internally.
- {py:func}`combra.image.bresenham_line` and friends — the geometry kernels the edge-classifier calls.
