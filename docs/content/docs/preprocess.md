---
title: "Image Preprocessing"
weight: 9
---

The described tools are designed for processing SEM images.

To explicitly highlight WC/Co phase boundaries, the following algorithms are applied sequentially:

1) A side of the image is selected. By default, the backscattered electron image is used.

2) The image is smoothed with a median filter to suppress noise and equalize pixel color distribution.

3) The smoothed image is binarized using Otsu's method.

4) A gradient is taken from the binarized image. It clearly shows WC/Co transitions.

5) The binarized image is inverted and a gradient map obtained in step 4 is added to it.

The complete image processing looks as follows:

```
preproc_image=1-Otsu(median_filter(image))+grad(Otsu(median_filter(image)))
```

Pixel values by classes:

* 0 - WC grain

* 127 - Co region

* 254 - boundary of Co region adjacent to WC grains. Boundary thickness - 1 pixel

## Processing a single image

```python
img=io.imread(img_path)
img=grainPreprocess.image_preprocess(img,h,k)
```

## Processing entire image dataset

```python
all_images=grainPreprocess.read_preprocess_data(images_folder_path, images_num_per_class=150,  preprocess=True, save=True)
```

## Image location

The location of source images and preprocessed images should look as follows:

```
project
│
└───images_folder
   │
   └───class1_images
   │       image1
   │       image2
   │       ...
   └───class2_images
   │       image1
   │       image2
   │       ...
   └───class3_images
   │       image1
   │       image2
   │       ...
```

## grainPreprocess class

### imdivide(image, h, side)

Divides the input image in half and returns the left or right part

**Parameters:**
- `image`: ndarray (height,width,channels)
- `h`: int scalar
- `side`: float scalar

**Returns:** ndarray (height,width/2,channels)

### combine(image, h, k=0.5)

Overlays left and right parts of the image. If k=1, the output will be the left part of the image, if k=0, it will be the right part

**Parameters:**
- `image`: ndarray (height,width,channels)
- `h`: int scalar
- `k`: float scalar

**Returns:** ndarray (height,width/2,channels)

### do_otsu(image)

Otsu binarization

**Parameters:**
- `img`: ndarray (height,width,channels)

**Returns:** ndarray (height,width), Boolean

### image_preprocess(image)

Combination of median filter, binarization and gradient. For grains, pixel value is 0, for binder regions - 127, and for their boundary - 254. The processed image is obtained as follows: 1-Otsu(median_filter(image))+grad(Otsu(median_filter(image)))

**Parameters:**
- `img`: ndarray (height,width,channels)

**Returns:** ndarray (height,width,1)

### read_preprocess_data(images_dir, max_images_num_per_class=100, preprocess=False, save=False, crop_bottom=False, h=135, resize=True, resize_shape=None, save_name='all_images.npy')

Reading entire dataset, processing and saving to .npy file

**Parameters:**
- `images_dir`: str
- `max_images_num_per_class`: int
- `preprocess`: Bool
- `save`: Bool
- `crop_bottom`: Bool
- `h`: int
- `resize`: Bool
- `resize_shape`: tuple (width, height, channels)
- `save_name`: str

**Returns:** ndarray (n_classes, n_images_per_class, width, height, channels)

### tiff2jpg(folder_path, start_name=0, stop_name=-4, new_folder_path='resized')

Converts from 16-bit tiff to 8-bit jpg

**Parameters:**
- `folder_path`: str
- `start_name`: int
- `stop_name`: int
- `new_folder_path`: str

**Returns:** None

### get_example_images()

Downloads 1 image of each sample from s3 container

**Returns:** ndarray [[img1],[img2]..]

### image_preprocess_kmeans(image, h=135, k=1, n_clusters=3, pos=1)

Boundary extraction using clustering and noise smoothing with median filter. Suitable only for blurred photographs where object boundaries are sufficiently wide. Pos controls the selection of the cluster that will be displayed on the returned image

**Parameters:**
- `image`: array (height,width,channels)
- `h`: int scalar
- `k`: float scalar
- `n_clusters`: int scalar
- `pos`: int scalar, cluster index

**Returns:** ndarray (height,width)
