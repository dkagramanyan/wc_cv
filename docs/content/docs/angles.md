---
title: "Angles and their distributions"
weight: 2
---

### Quick start

```python
from combra import angles, data
import json


step = 5

images_path = data.example_class_path()
json_save_name = 'test'

types_dict = {'Ultra_Co11': 'средние зерна',
              'Ultra_Co25': 'мелкие зерна',
              'Ultra_Co8': 'средне-мелкие зерна',
              'Ultra_Co6_2': 'крупные зерна',
              'Ultra_Co15': 'средне-мелкие зерна'}

angles.angles_approx_save(
                    images_path=images_path,
                    save_path=json_save_name,
                    types_dict=types_dict,
                    step=step,
                    max_images_num_per_class=360, 
                    workers = 20
                )

data = open(json_save_name+f'_step_{step}_degrees.json', encoding='utf-8')
data = json.load(data)
```

Angles distribution

```python

angles.angles_plot_base(data, plot_file_name=json_save_name, step=step, N=10, M=10, indices=[2,0,1], save=False)
```


### `get_angles()`

Main function for computing angles on a preprocessed image.

```python
from combra import angles, image
from skimage import io

# Image preprocessing
img = io.imread('image.png')
processed = image.image_preprocess(img)

# Compute angles
angles_array, contours = angles.get_angles(
    processed,
    border_eps=5,  # Distance from image edge
    tol=3           # Contour simplification accuracy
)

print(f"Found angles: {len(angles_array)}")
print(f"Mean angle: {angles_array.mean():.2f}°")
print(f"Median angle: {np.median(angles_array):.2f}°")
```

**Parameters:**
- `image` (ndarray): Preprocessed image (width, height, 1)
- `border_eps` (int): Distance from image edge to inner edge (contours too close to edge are ignored)
- `tol` (int): Contour simplification accuracy using Douglas-Peucker algorithm

**Returns:**
- `angles` (ndarray): Array of angles in degrees
- `contours` (list): List of processed contours

**Usage example:**
```python
import numpy as np
import matplotlib.pyplot as plt
from combra import angles, image, stats, approx
from skimage import io

# Load and preprocess
img = io.imread('grain_image.png')
processed = image.image_preprocess(img)

# Compute angles
angles_array, cnts = angles.get_angles(processed, border_eps=5, tol=3)

# Statistics
print(f"Total angles: {len(angles_array)}")
print(f"Min angle: {angles_array.min():.2f}°")
print(f"Max angle: {angles_array.max():.2f}°")
print(f"Mean angle: {angles_array.mean():.2f}°")
print(f"Standard deviation: {angles_array.std():.2f}°")

# Histogram
plt.hist(angles_array, bins=36, edgecolor='black')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.title('Angle Distribution')
plt.show()
```

## Processing entire dataset

### `angles_approx_save()`

Computes and saves angle distributions for all images in the dataset with automatic approximation.

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

# Compute and save
angles.angles_approx_save(
    images_path=images_path,
    save_path='angles_results',
    types_dict=types_dict,
    step=5,                      # Angle step in degrees
    max_images_num_per_class=360,
    workers=20                   # Number of processes
)
```

**Parameters:**
- `images_path` (str): Path to folder with image classes
- `save_path` (str): Base name for saving JSON file
- `types_dict` (dict): Dictionary mapping class names to their descriptions
- `step` (int): Angle step in degrees for histogram
- `max_images_num_per_class` (int, optional): Maximum images per class
- `workers` (int, optional): Number of processes for parallel processing

**Result:**
Creates a JSON file named `{save_path}_step_{step}_degrees.json` containing:
- Image paths
- Class names
- Text legends with distribution parameters
- Data for visualization (histograms)
- Gaussian approximation parameters

**JSON file structure:**
```json
[
  {
    "path": "/path/to/class",
    "name": "Ultra_Co11",
    "type": "medium grains",
    "legend": "--------------\nUltra_Co11 medium grains\n number of angles 1000\n...",
    "density_curve_scatter": [[0, 5, 10, ...], [10, 15, 20, ...]],
    "gauss_approx_plot": [[0, 1, 2, ...], [5.2, 5.5, 5.8, ...]],
    "gauss_approx_data": {
      "mus": [90.5, 270.3],
      "sigmas": [30.2, 28.7],
      "amps": [1.0, 0.95]
    }
  },
  ...
]
```

## Loading and analyzing results

```python
import json

# Load results
with open('angles_results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Analyze each class
for item in data:
    print(f"\nClass: {item['name']}")
    print(f"Type: {item['type']}")
    print(f"Legend:\n{item['legend']}")
    
    # Visualization data
    x, y = item['density_curve_scatter']
    print(f"Points in distribution: {len(x)}")
    
    # Approximation parameters
    mus = item['gauss_approx_data']['mus']
    sigmas = item['gauss_approx_data']['sigmas']
    amps = item['gauss_approx_data']['amps']
    
    print(f"Mode 1: μ={mus[0]:.2f}°, σ={sigmas[0]:.2f}°")
    print(f"Mode 2: μ={mus[1]:.2f}°, σ={sigmas[1]:.2f}°")
```

## Visualization

### `angles_plot_base()`

Visualization of angle distributions for multiple classes.

```python
from combra import angles
import json

# Load data
with open('angles_results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Visualization
angles.angles_plot_base(
    data,
    plot_file_name='angles_plot',
    step=5,
    N=10,              # Number of rows in grid
    M=10,              # Number of columns in grid
    indices=[0, 1, 2], # Class indices to display
    save=True,
    font_size=20,
    scatter_size=20
)
```

**Parameters:**
- `data`: Data from JSON file
- `plot_file_name`: Base name for saving plots
- `step`: Angle step in degrees
- `N`, `M`: Grid dimensions for plots
- `indices`: List of class indices to display (if `None`, all are displayed)
- `save`: Save plots
- `font_size`: Font size
- `scatter_size`: Point size on plot

### `angles_approx_modes()`

Visualization of angle distribution modes.

```python
from combra import angles

angles.angles_approx_modes(
    folder='results',      # Folder with JSON files
    step=5,
    start1=0, stop1=180,   # Range of first mode
    start2=180, stop2=360, # Range of second mode
    width=10,
    height=10,
    font_size=25
)
```

## Statistical analysis

### Integration with stats module

```python
from combra import angles, stats, approx
import matplotlib.pyplot as plt
import numpy as np

# Compute angles
angles_array, _ = angles.get_angles(processed, border_eps=5, tol=3)

# Preprocessing for statistics
x, y = stats.stats_preprocess(angles_array, step=5)

# Bimodal Gaussian approximation
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'o', label='Data', markersize=4, alpha=0.6)
plt.plot(x_gauss, y_gauss, '-', label='Bimodal approximation', linewidth=2)
plt.xlabel('Angle (degrees)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Angle Distribution with Approximation', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

print(f"Mode 1: μ={mus[0]:.2f}°, σ={sigmas[0]:.2f}°, A={amps[0]:.2f}")
print(f"Mode 2: μ={mus[1]:.2f}°, σ={sigmas[1]:.2f}°, A={amps[1]:.2f}")
```

## Creating legend

### `angles_legend()`

Creates a text legend for angle distribution.

```python
from combra import angles

legend = angles.angles_legend(
    images_amount=100,      # Number of images
    name='Ultra_Co11',      # Class name
    itype='medium grains',  # Grain type
    step=5,                 # Angle step
    mus=[90.5, 270.3],      # Mode mean values
    sigmas=[30.2, 28.7],    # Standard deviations
    amps=[1.0, 0.95],       # Amplitudes
    norm=1000               # Total number of angles
)

print(legend)
```

## Complete example

```python
from combra import angles, image, data, stats, approx
from skimage import io
import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Process single image
img = io.imread('grain_image.png')
processed = image.image_preprocess(img)
angles_array, cnts = angles.get_angles(processed, border_eps=5, tol=3)

# 2. Statistical analysis
x, y = stats.stats_preprocess(angles_array, step=5)
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# 3. Visualization
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'o', label='Data', markersize=4)
plt.plot(x_gauss, y_gauss, '-', label='Approximation', linewidth=2)
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 4. Process entire dataset
images_path = data.example_class_path()
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

# 5. Load and visualize results
with open('results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

angles.angles_plot_base(
    results,
    plot_file_name='angles_plot',
    step=5,
    N=10,
    M=10,
    save=True
)
```

## Parameter recommendations

### `border_eps`
- **Small values (3-5)**: More angles, but may have artifacts on edges
- **Large values (10-15)**: Fewer angles, but more reliable results

### `tol` (simplification accuracy)
- **Small values (1-3)**: More accurate contours, more points, slower
- **Large values (5-10)**: Fewer points, faster, but may lose detail

### `step` (histogram step)
- **5 degrees**: Standard value, good balance
- **1-2 degrees**: High detail, more noise
- **10 degrees**: Less detail, smoother distributions

## Notes

- Angles are computed taking into account contour traversal direction (counterclockwise)
- Angles >180° are taken into account (full range 0-360°)
- Contours too close to image edge are ignored
- For large datasets, parallel processing is recommended (`workers` parameter)
