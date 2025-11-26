---
title: "Crack Graph"
weight: 3
---

The `graph` module provides functionality for constructing and analyzing crack graphs on composite alloy images.

## Overview

A crack graph is a directed graph where:
- **Nodes** correspond to intersection or branching points of cracks
- **Edges** correspond to crack segments between nodes
- **Edge types** are classified by material: Co, WC-Co, WC, WC-WC

## Work pipeline

1. **Image preprocessing** (`image.image_preprocess`)
2. **Graph preprocessing** (`graph.preprocess_graph_image`) - contour extraction, node metadata preparation
3. **Graph construction** (`graph.create_crack_graph`) - creating a directed graph
4. **Energy computation** (optional) (`graph.get_energies`) - path energy calculation
5. **Visualization** (`graph.graph_plot`) - displaying graph on image

## Minimal example

```python
from combra import graph, image
from skimage import io

# 1. Load and preprocess image
img = io.imread('path/to/crack_image.png')
processed = image.image_preprocess(img)

# 2. Preprocess for graph
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        processed,
        r=2,
        border=30,
        border_node_eps=10,
        tol=5
    )
)

# 3. Create graph
G, img_contours_mono = graph.create_crack_graph(
    img_contours.shape,
    cnts,
    nodes_meta
)

print(f"Nodes in graph: {G.number_of_nodes()}")
print(f"Edges in graph: {G.number_of_edges()}")

# 4. Visualization
graph.graph_plot(
    G,
    img_contours_mono,
    N=10,
    M=10,
    name='crack_graph.png',
    save=True
)
```

## Graph preprocessing

### `preprocess_graph_image()`

Prepares image for graph construction: extracts contours, finds nodes, determines entry and exit nodes.

```python
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        image,
        r=2,                    # Radius for processing
        border=30,              # Border size
        border_node_eps=10,     # Epsilon for border nodes
        tol=5,                  # Contour simplification accuracy
        disk=5,                 # Disk size for morphology
        labeled_cnts=False,     # Use labeled contours
        labels=False            # Use labels
    )
)
```

**Parameters:**
- `image` (ndarray): Preprocessed image
- `r` (int): Radius for node processing
- `border` (int): Border size in pixels
- `border_node_eps` (int): Threshold for determining border nodes
- `tol` (int): Contour simplification accuracy (Douglas-Peucker algorithm)
- `disk` (int): Disk size for morphological operations
- `labeled_cnts` (bool): Use labeled contours
- `labels` (bool): Use class labels

**Returns:**
- `entry_nodes`: List of entry node indices
- `exit_nodes`: List of exit node indices
- `img_contours`: Image with contours
- `img_marked`: Labeled image
- `cnts`: List of contours
- `nodes_meta`: Dictionary with node metadata

## Graph construction

### `create_crack_graph()`

Creates a directed crack graph based on contours and node metadata.

```python
G, img_contours_mono = graph.create_crack_graph(
    img_shape,
    cnts,
    nodes_metadata,
    # Additional parameters...
)
```

**Parameters:**
- `img_shape`: Image shape (tuple)
- `cnts`: List of contours
- `nodes_metadata`: Dictionary with node metadata

**Returns:**
- `G`: NetworkX directed graph
- `img_contours_mono`: Monochrome contour image

**Graph structure:**
- Nodes contain attributes: coordinates, node type
- Edges contain attributes: edge type, length, energy (if computed)

## Edge types

### `get_edge_type()`

Determines edge type between two nodes based on pixel analysis along the line.

```python
edge_type = graph.get_edge_type(
    node1,
    node2,
    cnts,
    nodes_metadata,
    wc_eps=30,
    border_pixel=0
)
```

**Edge types:**
- `0`: Co (cobalt)
- `1`: WC-Co (tungsten-cobalt boundary)
- `2`: WC (tungsten carbide)
- `3`: WC-WC (tungsten-tungsten boundary)

### `get_edge_type_labeled()`

Edge type determination for labeled data.

```python
edge_type = graph.get_edge_type_labeled(
    node1,
    node2,
    nodes_metadata,
    line_eps=10
)
```

## Energy computation

### `get_energies()`

Computes path energies in the graph for various configurations.

```python
energies = graph.get_energies(
    g,
    cnts,
    nodes_metadata,
    entry_nodes,
    exit_nodes,
    workers=23
)
```

**Parameters:**
- `g`: NetworkX graph
- `cnts`: List of contours
- `nodes_metadata`: Node metadata
- `entry_nodes`: Entry nodes
- `exit_nodes`: Exit nodes
- `workers`: Number of processes

**Returns:**
- Dictionary with energies for various configurations

### `find_shortest_energy_paths()`

Finds shortest paths in the graph based on computed energies.

```python
paths = graph.find_shortest_energy_paths(
    g,
    energies_paths,
    ...
)
```

## Visualization

### `graph_plot()`

Visualizes graph on contour image.

```python
graph.graph_plot(
    g,
    img_contours,
    N=50,              # Number of rows in grid
    M=50,              # Number of columns in grid
    name='graph.jpg',  # Filename for saving
    border=30,         # Border size
    save=False         # Save image
)
```

**Example:**
```python
graph.graph_plot(
    G,
    img_contours_mono,
    N=10,
    M=10,
    name='crack_graph.png',
    border=30,
    save=True
)
```

### `plot_optimized_energies()`

Visualizes energy matrices for various configurations.

```python
graph.plot_optimized_energies(
    energies,
    ...
)
```

### `plot_paths()`

Visualizes paths on aligned image.

```python
graph.plot_paths(
    g,
    df,           # DataFrame with paths
    img_aligned,  # Aligned image
    border=30
)
```

### `plot_optimized_paths()`

Visualizes optimized paths.

```python
graph.plot_optimized_paths(
    g,
    energies_paths,
    img_contours_o,
    param_1=10,
    param_2=10
)
```

### `draw_tree()`

Visualizes tree on image.

```python
graph.draw_tree(
    img,
    centres=False,  # Show centers
    leafs=False,    # Show leaves
    nodes=False,    # Show nodes
    bones=False     # Show bones
)
```

## Complete example

```python
from combra import graph, image, contours
from skimage import io
import networkx as nx

# 1. Load and preprocess
img = io.imread('crack_image.png')
processed = image.image_preprocess(img)

# 2. Preprocess for graph
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        processed,
        r=2,
        border=30,
        border_node_eps=10,
        tol=5
    )
)

# 3. Create graph
G, img_contours_mono = graph.create_crack_graph(
    img_contours.shape,
    cnts,
    nodes_meta
)

# 4. Graph analysis
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Entry nodes: {len(entry_nodes)}")
print(f"Exit nodes: {len(exit_nodes)}")

# 5. Energy computation (optional)
energies = graph.get_energies(
    G,
    cnts,
    nodes_meta,
    entry_nodes,
    exit_nodes,
    workers=4
)

# 6. Visualization
graph.graph_plot(
    G,
    img_contours_mono,
    N=10,
    M=10,
    name='crack_graph.png',
    save=True
)

# 7. Edge type analysis
edge_types = {}
for u, v, data in G.edges(data=True):
    edge_type = data.get('type', 'unknown')
    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

print("Edge type distribution:")
for edge_type, count in edge_types.items():
    type_names = {0: 'Co', 1: 'WC-Co', 2: 'WC', 3: 'WC-WC'}
    print(f"  {type_names.get(edge_type, 'Unknown')}: {count}")
```

## Working with NetworkX

Since the graph is a NetworkX object, you can use all standard NetworkX functions:

```python
import networkx as nx

# Find shortest paths
if nx.has_path(G, source=entry_nodes[0], target=exit_nodes[0]):
    path = nx.shortest_path(G, source=entry_nodes[0], target=exit_nodes[0])
    print(f"Shortest path: {path}")

# Compute centrality
centrality = nx.degree_centrality(G)
print(f"Most central node: {max(centrality, key=centrality.get)}")

# Find connected components
components = list(nx.weakly_connected_components(G))
print(f"Connected components: {len(components)}")
```

## Parameters and settings

### Recommended parameters

For most cases, the following parameters work well:

```python
# Preprocessing
processed = image.image_preprocess(img)

# Graph preprocessing
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        processed,
        r=2,                # Standard value
        border=30,          # Depends on image size
        border_node_eps=10, # Standard value
        tol=5,              # Balance between accuracy and performance
        disk=5              # Standard value
    )
)
```

### Tuning for different image types

- **High resolution**: increase `border` and `border_node_eps`
- **High noise**: increase `disk` for morphology
- **Complex contours**: decrease `tol` for greater accuracy

## Notes

- Graph is directed
- Nodes on image border are automatically determined as entry/exit
- Edge types are determined based on pixel analysis along the line between nodes
- For large images, parallel processing is recommended (`workers` parameter)
