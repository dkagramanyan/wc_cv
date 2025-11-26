---
title: "Quick Start"
weight: 1
---

Below are minimal examples of using the main functions of the `combra` package.

## Installation

```bash
pip install -e .
```

## Preprocessing a single image

```python
from combra import image
from skimage import io

image_raw = io.imread('path/to/image.png')
processed = image.image_preprocess(image_raw)
```

The `image_preprocess` function performs:
- Median filtering
- Otsu binarization
- Gradient computation

## Contour extraction

```python
from combra import contours

cnts = contours.get_contours(processed, tol=3)
print(f"Found contours: {len(cnts)}")
```

## Angle computation

```python
from combra import angles

angles_array, contours = angles.get_angles(processed, border_eps=5, tol=3)
print(f"Found angles: {len(angles_array)}")
print(f"Mean angle: {angles_array.mean():.2f}°")
```

## Working with dataset

```python
from combra.data.dataset import SEMDataset

# Create dataset (automatically caches preprocessed images in /tmp)
dataset = SEMDataset('data/wc_co', max_images_num_per_class=50)

# Get image and path
image, path = dataset.__getitem__(0, 0)
print(f"Loaded image: {path}")

# Number of classes
print(f"Classes in dataset: {len(dataset)}")
```

## Crack graph

### Creating graph

```python
from combra import graph, image
from skimage import io

# Image preprocessing
img = io.imread('crack_image.png')
processed = image.image_preprocess(img)

# Preprocessing for graph
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(processed, tol=5)
)

# Create graph
G, img_contours_mono = graph.create_crack_graph(
    img_contours.shape,
    cnts,
    nodes_meta
)

print(f"Nodes in graph: {G.number_of_nodes()}")
print(f"Edges in graph: {G.number_of_edges()}")
```

### Graph visualization

```python
from combra import graph

graph.graph_plot(
    G,
    img_contours,
    N=10,
    M=10,
    name='crack_graph.png',
    save=True
)
```

## Contour visualization

```python
from combra import contours
from PIL import Image
import numpy as np

# Create RGB image
img_rgb = Image.fromarray(np.zeros_like(processed, dtype=np.uint8).repeat(3, axis=2))

# Draw contours
img_with_contours = contours.draw_contours(
    img_rgb,
    cnts=cnts,
    color_corner=(0, 139, 139),
    color_line=(255, 140, 0),
    corners=True
)
```

## Processing entire angle dataset

```python
from combra import angles, data
import json

# Path to dataset
images_path = data.example_class_path()

# Dictionary of grain types
types_dict = {
    'Ultra_Co11': 'medium grains',
    'Ultra_Co25': 'fine grains',
    'Ultra_Co8': 'medium-fine grains',
    'Ultra_Co6_2': 'coarse grains',
    'Ultra_Co15': 'medium-fine grains'
}

# Compute and save angle distributions
angles.angles_approx_save(
    images_path=images_path,
    save_path='angles_results',
    types_dict=types_dict,
    step=5,  # Step in degrees
    max_images_num_per_class=360,
    workers=20
)

# Load results
with open('angles_results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Visualization
angles.angles_plot_base(
    results,
    plot_file_name='angles_plot',
    step=5,
    N=10,
    M=10,
    indices=[0, 1, 2],  # Class indices to display
    save=True
)
```

## Statistical analysis and approximation

```python
from combra import stats, approx
import matplotlib.pyplot as plt
import numpy as np

# Data preprocessing
x, y = stats.stats_preprocess(angles_array, step=5)

# Bimodal Gaussian approximation
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Data', markersize=4)
plt.plot(x_gauss, y_gauss, '-', label='Approximation', linewidth=2)
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.title('Angle Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Mode 1: μ={mus[0]:.2f}°, σ={sigmas[0]:.2f}°")
print(f"Mode 2: μ={mus[1]:.2f}°, σ={sigmas[1]:.2f}°")
```

## MVEE (Minimum Volume Enclosing Ellipsoid)

```python
from combra import mvee, data

images_path = data.example_class_path()
types_dict = {
    'Ultra_Co11': 'medium grains',
    'Ultra_Co25': 'fine grains'
}

# Compute MVEE for dataset
mvee.diametr_approx_save(
    images_path=images_path,
    save_path='mvee_results',
    types_dict=types_dict,
    step=4,
    pixel=50/1000,  # Pixel size in mm
    max_images_num_per_class=None
)

# Load and visualize
import json
with open('mvee_results_step_4_beams.json', 'r') as f:
    mvee_data = json.load(f)

mvee.plot_beam_base(
    mvee_data,
    save_name='mvee_plot',
    step=4,
    N=7,
    M=7,
    save=True
)
```

## Next steps

For more detailed information see:
- [Dataset: SEMDataset](/docs/usage_dataset) - more about working with datasets
- [Crack Graph](/docs/usage_graph) - graph construction details
- [Angles](/docs/usage_angles) - angle analysis and distributions
- [API Reference](/docs/api) - complete API reference
