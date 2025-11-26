---
title: "Dataset: SEMDataset"
weight: 2
---

The `SEMDataset` class prepares a cache of preprocessed images in `/tmp/<dataset_name>` and provides convenient access to images and their paths.

## Main features

- Automatic caching of preprocessed images
- Parallel processing using all available CPU cores
- Cache integrity checking and automatic update when needed
- Support for folder structure with image classes

## Creating dataset

```python
from combra.data.dataset import SEMDataset

# Create dataset
# Path should not end with '/'
dataset = SEMDataset(
    images_folder_path='data/wc_co',
    max_images_num_per_class=100,
    workers=None  # Default: cpu_count()-1
)
```

**Parameters:**

- `images_folder_path` (str): Root folder with class subfolders. **Important:** path should not end with `/`
- `max_images_num_per_class` (int): Image limit per class. If `None`, all available images are used
- `workers` (int, optional): Number of processes for parallel processing. Default is `cpu_count()-1`

## Folder structure

The dataset expects the following structure:

```
data/wc_co/
├── Ultra_Co11/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Ultra_Co25/
│   ├── image1.jpg
│   └── ...
└── Ultra_Co8/
    └── ...
```

## Data access

### Getting image

```python
# Get image and path by class and image indices
image, path = dataset.__getitem__(class_idx=0, idx=0)

print(f"Image path: {path}")
print(f"Image shape: {image.shape}")
```

### Number of classes

```python
# Number of classes in dataset
num_classes = len(dataset)
print(f"Classes in dataset: {num_classes}")
```

### Iterating over dataset

```python
# Iterate over all classes and images
for class_idx in range(len(dataset)):
    num_images = dataset.images_paths.shape[1]
    for img_idx in range(num_images):
        image, path = dataset.__getitem__(class_idx, img_idx)
        # Process image
        print(f"Class {class_idx}, Image {img_idx}: {path}")
```

## Image preprocessing

The class automatically applies preprocessing to each image:

1. Conversion to grayscale (if necessary)
2. Median filtering
3. Otsu binarization
4. Gradient computation

### Using preprocessing method separately

```python
from combra.data.dataset import SEMDataset
from skimage import io

# Load image
image = io.imread('path/to/image.jpg')

# Preprocessing
processed = SEMDataset.preprocess_image(
    image,
    pad=False,      # Add border
    border=30,       # Border size in pixels
    disk=3           # Disk size for median filter
)
```

## Caching

The dataset automatically caches preprocessed images in `/tmp/<dataset_name>`. 

### Cache checking

When creating the dataset, the following checks are performed:
- Cache existence
- Folder structure match
- Number of images

If the cache is incomplete or outdated, it is automatically recreated.

### Clearing cache

```python
import shutil
import os

cache_dir = '/tmp/wc_co'  # Replace with your dataset name
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cache deleted")
```

## Usage examples

### Basic example

```python
from combra.data.dataset import SEMDataset

# Create dataset
dataset = SEMDataset('data/wc_co', max_images_num_per_class=50)

# Get first image of first class
image, path = dataset.__getitem__(0, 0)
print(f"Loaded: {path}")
print(f"Size: {image.shape}")
```

### Processing all images

```python
from combra.data.dataset import SEMDataset
from combra import angles

dataset = SEMDataset('data/wc_co', max_images_num_per_class=100)

all_angles = []

for class_idx in range(len(dataset)):
    for img_idx in range(dataset.images_paths.shape[1]):
        image, path = dataset.__getitem__(class_idx, img_idx)
        
        # Compute angles
        angles_array, _ = angles.get_angles(image, border_eps=5, tol=3)
        all_angles.extend(angles_array)
        
        print(f"Processed: {path}")

print(f"Total angles: {len(all_angles)}")
```

### Using with other modules

```python
from combra.data.dataset import SEMDataset
from combra import angles, stats, approx
import matplotlib.pyplot as plt

# Create dataset
dataset = SEMDataset('data/wc_co', max_images_num_per_class=50)

# Collect angles from all images
all_angles = []
for class_idx in range(len(dataset)):
    for img_idx in range(dataset.images_paths.shape[1]):
        image, _ = dataset.__getitem__(class_idx, img_idx)
        angles_array, _ = angles.get_angles(image)
        all_angles.extend(angles_array)

# Statistical analysis
x, y = stats.stats_preprocess(all_angles, step=5)

# Approximation
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# Visualization
plt.plot(x, y, 'o', label='Data')
plt.plot(x_gauss, y_gauss, '-', label='Approximation')
plt.legend()
plt.show()
```

## Notes

- Dataset path **should not** end with `/`
- Cache is created once and reused on subsequent runs
- Preprocessed images are saved in PNG format
- All images should have the same size (check is not performed automatically)

## Class methods

### `__init__(images_folder_path, max_images_num_per_class=100, workers=None)`

Initialize dataset.

### `__getitem__(class_idx, idx)`

Get image and path.

**Parameters:**
- `class_idx` (int): Class index
- `idx` (int): Image index in class

**Returns:**
- `tuple`: `(image, path)` where `image` - numpy array, `path` - string with path

### `__len__()`

Returns the number of classes in the dataset.

### `preprocess_image(image, pad=False, border=30, disk=3)` (classmethod)

Static method for preprocessing a single image.

**Parameters:**
- `image` (ndarray): Input image
- `pad` (bool): Add border
- `border` (int): Border size
- `disk` (int): Disk size for median filter

**Returns:**
- `ndarray`: Preprocessed image
