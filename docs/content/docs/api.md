---
title: "API Reference"
weight: 21
---

# API Reference

Complete API reference for the `combra` library - computer vision for composite alloys.

## Modules

The `combra` library is organized into the following modules:

- [`data`](#data) - Working with datasets and example data
- [`image`](#image) - Image preprocessing
- [`contours`](#contours) - Contour extraction and processing
- [`angles`](#angles) - Angle computation and their distributions
- [`graph`](#graph) - Crack graph construction and analysis
- [`mvee`](#mvee) - Minimum Volume Enclosing Ellipsoid (MVEE)
- [`areas`](#areas) - Polygon area analysis
- [`approx`](#approx) - Data approximation
- [`stats`](#stats) - Statistical processing and distributions

---

## data

Module for working with image datasets.

### SEMDataset

Class for loading and preprocessing image datasets with automatic caching.

```python
from combra.data.dataset import SEMDataset

dataset = SEMDataset(
    images_folder_path='path/to/images',
    max_images_num_per_class=100,
    workers=None  # Default: cpu_count()-1
)

# Get image and path
image, path = dataset.__getitem__(class_idx=0, idx=0)

# Number of classes
num_classes = len(dataset)
```

**Parameters:**
- `images_folder_path` (str): Path to root folder with class subfolders (should not end with `/`)
- `max_images_num_per_class` (int): Maximum number of images per class
- `workers` (int, optional): Number of processes for parallel processing

**Methods:**
- `__getitem__(class_idx, idx)`: Returns tuple `(image, path)` for specified class and index
- `__len__()`: Returns number of classes
- `preprocess_image(image, pad=False, border=30, disk=3)`: Static method for preprocessing a single image

**Example:**
```python
from combra.data.dataset import SEMDataset

# Create dataset (automatically caches in /tmp)
dataset = SEMDataset('data/wc_co', max_images_num_per_class=50)

# Iterate over all classes and images
for class_idx in range(len(dataset)):
    for img_idx in range(dataset.images_paths.shape[1]):
        image, path = dataset.__getitem__(class_idx, img_idx)
        # Process image
        print(f"Class {class_idx}, Image {img_idx}: {path}")
```

### example_images()

Returns a list of example images from the built-in dataset.

```python
from combra import data

images = data.example_images()
# Returns: List[Tuple[str, np.ndarray]]
```

### example_class_path()

Returns path to folder with example classes.

```python
from combra import data

path = data.example_class_path()
# Returns: str
```

### example_crack_fixed_images()

Returns a list of example crack images.

```python
from combra import data

images = data.example_crack_fixed_images()
# Returns: List[Tuple[str, np.ndarray]]
```

---

## image

Module for image preprocessing.

### image_preprocess(image)

Main image preprocessing function: median filter, Otsu binarization and gradient.

```python
from combra import image
from skimage import io

img = io.imread('path/to/image.png')
processed = image.image_preprocess(img)
```

**Parameters:**
- `image` (ndarray): Input image (height, width, channels)

**Returns:**
- `ndarray`: Preprocessed image (height, width, 1)

### preprocess_image(image, pad=False, border=30, disk=3)

Image preprocessing with optional border addition.

```python
from combra.data.dataset import SEMDataset

processed = SEMDataset.preprocess_image(
    image,
    pad=True,
    border=30,
    disk=3
)
```

**Parameters:**
- `image` (ndarray): Input image
- `pad` (bool): Add border
- `border` (int): Border size in pixels
- `disk` (int): Disk size for median filter

### do_otsu(img)

Image binarization using Otsu method.

```python
from combra import image

binary = image.do_otsu(img)
```

### align_figures(orig_img_padded, tol, labeled_cnts=False, labels=False)

Figure alignment on image.

```python
from combra import image

aligned = image.align_figures(
    orig_img_padded,
    tol=5,
    labeled_cnts=False
)
```

### fill_polygon(grid, corners, fill_value=1)

Fill polygon on grid.

```python
from combra import image

filled = image.fill_polygon(grid, corners, fill_value=1)
```

### get_perp_v(start_node_x, start_node_y, end_node_x, end_node_y, line_eps=10)

Compute perpendicular vector to line.

```python
from combra import image

perp_vector = image.get_perp_v(x1, y1, x2, y2, line_eps=10)
```

### find_intersection_2d(p1, p2, p3, p4)

Find intersection point of two line segments in 2D.

```python
from combra import image

intersection = image.find_intersection_2d(p1, p2, p3, p4)
```

### is_point_in_polygon(x, y, corners)

Check if point is inside polygon.

```python
from combra import image

inside = image.is_point_in_polygon(x, y, corners)
```

### get_bresenham_eps_pixels(img_contours_np, start_node_x, start_node_y, end_node_x, end_node_y, border_pixel=255)

Get pixels along Bresenham line with epsilon neighborhood.

```python
from combra import image

pixels = image.get_bresenham_eps_pixels(
    img_contours_np,
    x1, y1, x2, y2,
    border_pixel=255
)
```

---

## contours

Module for working with contours.

### get_row_contours(image)

Extract raw contours from image.

```python
from combra import contours

cnts = contours.get_row_contours(image)
# Returns: list of ndarray, each element is an array of contour point coordinates
```

**Parameters:**
- `image` (ndarray): Input image (width, height, 3)

**Returns:**
- `list`: List of contours, each contour is an array of shape (M_points, 2)

### get_contours(image, tol=3)

Extract contours with Douglas-Peucker algorithm simplification.

```python
from combra import contours

cnts = contours.get_contours(image, tol=3)
```

**Parameters:**
- `image` (ndarray): Input image
- `tol` (int): Maximum distance from original points to simplified polygonal chain

**Returns:**
- `list`: List of simplified contours

**Example:**
```python
from combra import image, contours
from skimage import io

# Preprocessing
img = io.imread('image.png')
processed = image.image_preprocess(img)

# Extract contours
cnts = contours.get_contours(processed, tol=3)
print(f"Found contours: {len(cnts)}")
```

### draw_contours(image, cnts, color_corner=(0, 139, 139), color_line=(255, 140, 0), r=2, e_width=5, l_width=2, corners=False)

Visualize contours on image.

```python
from combra import contours
from PIL import Image
import numpy as np

img_rgb = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
img_with_contours = contours.draw_contours(
    img_rgb,
    cnts,
    color_corner=(0, 139, 139),
    color_line=(255, 140, 0),
    corners=True
)
```

**Parameters:**
- `image`: PIL Image or ndarray
- `cnts`: List of contours
- `color_corner`: Corner color (RGB)
- `color_line`: Line color (RGB)
- `r`: Corner point radius
- `e_width`: Contour line width
- `l_width`: Line width
- `corners`: Show corners

### draw_edges(image, cnts, color=(0, 139, 139), r=4, e_width=5, l_width=4)

Visualize contour edges.

```python
from combra import contours

img_with_edges = contours.draw_edges(image, cnts, color=(0, 139, 139))
```

### mark_corners_and_classes(image, max_num=100000, sens=0.1, max_dist=1)

Corner detection and classification by gradients.

```python
from combra import contours

corners, classes, num = contours.mark_corners_and_classes(
    image,
    max_num=100000,
    sens=0.1,
    max_dist=1
)
```

### skeletons_coords(image)

Extract skeleton coordinates from binarized image.

```python
from combra import contours

bones = contours.skeletons_coords(binary_image)
```

---

## angles

Module for computing angles and analyzing their distributions.

### get_angles(image, border_eps=5, tol=3)

Compute angles on image contours.

```python
from combra import angles
from combra import image

processed = image.image_preprocess(img)
angles_array, contours = angles.get_angles(processed, border_eps=5, tol=3)
```

**Parameters:**
- `image` (ndarray): Preprocessed image (width, height, 1)
- `border_eps` (int): Distance from image edge to inner edge
- `tol` (int): Contour simplification accuracy

**Returns:**
- `angles` (ndarray): Array of angles in degrees
- `contours` (list): List of processed contours

**Example:**
```python
from combra import angles, image, data
from skimage import io

# Load and preprocess
img = io.imread('image.png')
processed = image.image_preprocess(img)

# Compute angles
angles_array, cnts = angles.get_angles(processed, border_eps=5, tol=3)
print(f"Found angles: {len(angles_array)}")
print(f"Mean angle: {angles_array.mean():.2f}°")
```

### angles_approx_save(images_path, save_path, types_dict, step, max_images_num_per_class=None, workers=None)

Compute and save angle distribution for all images in dataset.

```python
from combra import angles, data

images_path = data.example_class_path()
types_dict = {
    'Ultra_Co11': 'medium grains',
    'Ultra_Co25': 'fine grains',
    'Ultra_Co8': 'medium-fine grains'
}

angles.angles_approx_save(
    images_path=images_path,
    save_path='angles_results',
    types_dict=types_dict,
    step=5,  # Step in degrees
    max_images_num_per_class=360,
    workers=20
)
```

**Parameters:**
- `images_path` (str): Path to folder with image classes
- `save_path` (str): Base name for saving JSON file
- `types_dict` (dict): Dictionary mapping class names to their descriptions
- `step` (int): Angle step in degrees for histogram
- `max_images_num_per_class` (int, optional): Maximum images per class
- `workers` (int, optional): Number of processes

**Returns:**
- Saves JSON file named `{save_path}_step_{step}_degrees.json`

**Example of using saved data:**
```python
import json

with open('angles_results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    print(f"Class: {item['name']}")
    print(f"Legend: {item['legend']}")
    x, y = item['density_curve_scatter']
    print(f"Points in distribution: {len(x)}")
```

### angles_plot_base(data, plot_file_name, step, N, M, save=False, indices=None, font_size=20, scatter_size=20)

Visualize angle distribution.

```python
from combra import angles
import json

with open('angles_results_step_5_degrees.json', 'r') as f:
    data = json.load(f)

angles.angles_plot_base(
    data,
    plot_file_name='angles_plot',
    step=5,
    N=10,  # Number of rows in grid
    M=10,  # Number of columns in grid
    indices=[0, 1, 2],  # Class indices to display
    save=True
)
```

### angles_approx_modes(folder, step, start1, stop1, start2, stop2, width, height, font_size=25)

Visualize angle distribution modes.

```python
from combra import angles

angles.angles_approx_modes(
    folder='results',
    step=5,
    start1=0, stop1=180,
    start2=180, stop2=360,
    width=10, height=10,
    font_size=25
)
```

### angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm)

Create text legend for angle distribution.

```python
from combra import angles

legend = angles.angles_legend(
    images_amount=100,
    name='Ultra_Co11',
    itype='medium grains',
    step=5,
    mus=[90, 270],
    sigmas=[30, 30],
    amps=[1.0, 1.0],
    norm=1000
)
```

---

## graph

Module for constructing and analyzing crack graphs.

### preprocess_graph_image(image, r=2, border=30, border_node_eps=10, tol=5, disk=5, labeled_cnts=False, labels=False, ...)

Preprocess image for crack graph construction.

```python
from combra import graph, image
from skimage import io

img = io.imread('crack_image.png')
processed = image.image_preprocess(img)

entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        processed,
        r=2,
        border=30,
        border_node_eps=10,
        tol=5
    )
)
```

**Returns:**
- `entry_nodes`: Graph entry nodes
- `exit_nodes`: Graph exit nodes
- `img_contours`: Image with contours
- `img_marked`: Labeled image
- `cnts`: List of contours
- `nodes_meta`: Node metadata

### create_crack_graph(img_shape, cnts, nodes_metadata, ...)

Create directed crack graph.

```python
from combra import graph
import networkx as nx

G, img_contours_mono = graph.create_crack_graph(
    img_contours.shape,
    cnts,
    nodes_meta
)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
```

**Returns:**
- `G`: NetworkX graph
- `img_contours_mono`: Monochrome contour image

### get_edge_type(node1, node2, cnts, nodes_metadata, wc_eps=30, border_pixel=0)

Determine edge type between nodes.

```python
from combra import graph

edge_type = graph.get_edge_type(
    node1, node2,
    cnts,
    nodes_metadata,
    wc_eps=30
)
# Returns: 0 (Co), 1 (WC-Co), 2 (WC), 3 (WC-WC)
```

### get_edge_type_labeled(node1, node2, nodes_metadata, line_eps=10)

Determine edge type for labeled data.

```python
from combra import graph

edge_type = graph.get_edge_type_labeled(
    node1, node2,
    nodes_metadata,
    line_eps=10
)
```

### get_energies(g, cnts, nodes_metadata, entry_nodes, exit_nodes, ...)

Compute path energies in graph.

```python
from combra import graph

energies = graph.get_energies(
    g,
    cnts,
    nodes_metadata,
    entry_nodes,
    exit_nodes,
    workers=23
)
```

### find_shortest_energy_paths(g, energies_paths, ...)

Find shortest paths by energies.

```python
from combra import graph

paths = graph.find_shortest_energy_paths(
    g,
    energies_paths,
    ...
)
```

### graph_plot(g, img_contours, N=50, M=50, name='graph.jpg', border=30, save=False)

Visualize graph on image.

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

### plot_optimized_energies(...)

Visualize energy matrices.

```python
from combra import graph

graph.plot_optimized_energies(
    energies,
    ...
)
```

### plot_paths(g, df, img_aligned, border=30)

Visualize paths on image.

```python
from combra import graph

graph.plot_paths(G, paths_df, img_aligned, border=30)
```

### plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10)

Visualize optimized paths.

```python
from combra import graph

graph.plot_optimized_paths(
    G,
    energies_paths,
    img_contours_o,
    param_1=10,
    param_2=10
)
```

### draw_tree(img, centres=False, leafs=False, nodes=False, bones=False)

Visualize tree on image.

```python
from combra import graph

graph.draw_tree(
    img,
    centres=True,
    leafs=True,
    nodes=True
)
```

---

## mvee

Module for working with Minimum Volume Enclosing Ellipsoid (MVEE).

### get_mvee_params(image, tol=0.2)

Compute MVEE parameters for image.

```python
from combra import mvee

params = mvee.get_mvee_params(image, tol=0.2)
```

### diametr_approx_save(images_path, save_path, types_dict, step, pixel, start=2, end=-3, max_images_num_per_class=None)

Compute and save diameters (MVEE) for dataset.

```python
from combra import mvee, data

images_path = data.example_class_path()
types_dict = {
    'Ultra_Co11': 'medium grains',
    'Ultra_Co25': 'fine grains'
}

mvee.diametr_approx_save(
    images_path=images_path,
    save_path='mvee_results',
    types_dict=types_dict,
    step=4,
    pixel=50/1000,  # Pixel size in mm
    max_images_num_per_class=None
)
```

### plot_beam_base(data, save_name, step, N, M, indices=None, save=False, scatter_size=60, font_size=20)

Visualize MVEE base data.

```python
from combra import mvee
import json

with open('mvee_results_step_4_beams.json', 'r') as f:
    data = json.load(f)

mvee.plot_beam_base(
    data,
    save_name='mvee_plot',
    step=4,
    N=7,
    M=7,
    save=True
)
```

### plot_angles(data, saved_image_name, step, N, M, indices=None, save=False)

Visualize MVEE angles.

```python
from combra import mvee

mvee.plot_angles(data, 'angles_plot', step=4, N=7, M=7, save=True)
```

### beams_heatmap(data, step, saved_names, indices=None, bin_max=30, N=7, M=7, font_size=20, save=False, scatter_size=60)

Visualize MVEE heatmap.

```python
from combra import mvee

mvee.beams_heatmap(
    data,
    step=4,
    saved_names=['heatmap'],
    N=7,
    M=7,
    save=True
)
```

### enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)

Visualize enclosing ellipse.

```python
from combra import mvee

mvee.enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)
```

---

## areas

Module for polygon area analysis.

### plot_polygons_area(data, saved_image_name, step, N, M, indices=None, save=False, start=1, end=None, pixel=50/1000, font_size=20, s=60, log_min_val=-8, min_area_num=10)

Visualize polygon area distribution.

```python
from combra import areas
import json

with open('areas_data.json', 'r') as f:
    data = json.load(f)

areas.plot_polygons_area(
    data,
    saved_image_name='areas_plot',
    step=5,
    N=10,
    M=10,
    pixel=50/1000,  # Pixel size in mm
    save=True
)
```

### plot_polygons_effective_radius(data, saved_image_name, step, N, M, indices=None, save=False, start=1, end=None, ...)

Visualize polygon effective radii.

```python
from combra import areas

areas.plot_polygons_effective_radius(
    data,
    saved_image_name='radius_plot',
    step=5,
    N=10,
    M=10,
    save=True
)
```

---

## approx

Module for data approximation.

### gaussian_fit(x, y, mu=1, sigma=1, amp=1)

Approximate data with single Gaussian function.

```python
from combra import approx
import numpy as np

x = np.linspace(0, 360, 100)
y = some_data  # Your data

mu, sigma, amp = approx.gaussian_fit(x, y, mu=100, sigma=30, amp=1)
print(f"μ={mu:.2f}, σ={sigma:.2f}, A={amp:.2f}")
```

**Returns:**
- `mu`: Mean value
- `sigma`: Standard deviation
- `amp`: Amplitude

### gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)

Approximate data with bimodal Gaussian function.

```python
from combra import approx

mus, sigmas, amps = approx.gaussian_fit_bimodal(
    x, y,
    mu1=90, mu2=270,
    sigma1=30, sigma2=30,
    amp1=1, amp2=1
)
```

**Returns:**
- `mus`: [mu1, mu2]
- `sigmas`: [sigma1, sigma2]
- `amps`: [amp1, amp2]

### gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1, amp3=1)

Approximate data with trimodal Gaussian function.

```python
from combra import approx

mus, sigmas, amps = approx.gaussian_fit_termodal(
    x, y,
    mu1=10, mu2=100, mu3=240,
    sigma1=10, sigma2=30, sigma3=30,
    amp1=1, amp2=1, amp3=1
)
```

### bimodal_gauss_approx(x, y)

Complete bimodal approximation with curve return.

```python
from combra import approx

(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# Visualization
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Data')
plt.plot(x_gauss, y_gauss, '-', label='Approximation')
plt.legend()
plt.show()
```

**Returns:**
- `(x_gauss, y_gauss)`: Approximating curve coordinates
- `mus`: [mu1, mu2]
- `sigmas`: [sigma1, sigma2]
- `amps`: [amp1, amp2]

### lin_regr_approx(x, y)

Linear regression.

```python
from combra import approx

slope, intercept = approx.lin_regr_approx(x, y)
```

---

## stats

Module for statistical processing.

### stats_preprocess(array, step)

Preprocess array for statistical analysis.

```python
from combra import stats

angles = np.array([...])  # Array of angles
x, y = stats.stats_preprocess(angles, step=5)
# x - bin centers, y - frequencies
```

**Parameters:**
- `array` (ndarray): Input data array
- `step` (float): Step for histogram

**Returns:**
- `x`: Bin centers
- `y`: Frequencies

### gaussian(x, mu, sigma, amp=1)

Gaussian function.

```python
from combra import stats
import numpy as np

x = np.linspace(0, 360, 100)
y = stats.gaussian(x, mu=100, sigma=30, amp=1)
```

### gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1)

Bimodal Gaussian function.

```python
from combra import stats

y = stats.gaussian_bimodal(
    x,
    mu1=90, mu2=270,
    sigma1=30, sigma2=30,
    amp1=1, amp2=1
)
```

### gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1)

Trimodal Gaussian function.

```python
from combra import stats

y = stats.gaussian_termodal(
    x,
    mu1=10, mu2=100, mu3=240,
    sigma1=10, sigma2=30, sigma3=30,
    amp1=1, amp2=1, amp3=1
)
```

### ellipse(a, b, angle, xc=0, yc=0, num=50)

Generate ellipse coordinates.

```python
from combra import stats

coords = stats.ellipse(a=10, b=5, angle=45, xc=0, yc=0, num=50)
```

---

## Complete usage example

```python
from combra import data, image, angles, approx, stats
from skimage import io
import json

# 1. Load example data
images_path = data.example_class_path()

# 2. Process single image
img = io.imread('path/to/image.png')
processed = image.image_preprocess(img)

# 3. Compute angles
angles_array, contours = angles.get_angles(processed, border_eps=5, tol=3)

# 4. Statistical processing
x, y = stats.stats_preprocess(angles_array, step=5)

# 5. Approximation
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# 6. Visualization
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Data')
plt.plot(x_gauss, y_gauss, '-', label='Approximation')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 7. Process entire dataset
types_dict = {
    'Ultra_Co11': 'medium grains',
    'Ultra_Co25': 'fine grains'
}

angles.angles_approx_save(
    images_path=images_path,
    save_path='results',
    types_dict=types_dict,
    step=5,
    max_images_num_per_class=100
)
```
