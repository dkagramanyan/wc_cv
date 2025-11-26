---
title: "Installation"
weight: 6
---

## Installation from source

To work with the `combra` library, clone the repository and install the package in development mode:

```bash
git clone https://github.com/dkagramanyan/wc_cv
cd wc_cv/combra
pip install -e .
```

Required Python version - 3.8 or higher.

## Dependencies

Main dependencies are installed automatically when installing the package. Main libraries:
- `numpy` - array operations
- `scikit-image` - image processing
- `networkx` - graph operations
- `lmfit` - function approximation
- `opencv-python` - computer vision
- `mpire` - parallel processing

## Installation verification

```python
import combra
print(f"combra version: {combra.__version__}")

# Check main modules
from combra import data, image, angles, graph
print("All modules successfully imported!")
```

Test images are located in s3 storage at the following addresses:

1. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co11.jpg 
2. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co15.jpg
3. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co25.jpg
4. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co6_2.jpg
5. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co8.jpg
