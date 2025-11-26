---
title: "Mark"
weight: 12
---

## Determining angles at Co regions

To determine angles, three tools are used sequentially:

1) Canny edge detection. It finds all possible edges on the image and then highlights them.

2) Suzuki contour finding method. It is needed for labeling (numbering) pixels of contours found in step 1. Contour traversal direction is counterclockwise. Let's verify the direction. The point is the start of contour drawing.

![Suzuki](https://pobedit.s3.us-east-2.amazonaws.com/docs_images/suzuki.jpg)

3) The obtained contour pixel arrays are approximated using the Douglas-Peucker method. It leaves only those points from the point array that describe the polygon's character, for example: corner points, inflection points, etc.

As a result, for each contour we have an array of points that describe the contour perimeter with a given accuracy.

To determine the boundary, we use the determinant of three vectors. If the determinant det is less than 0, then angle phi is less than 180 degrees, otherwise greater than 180.

```
grainMark.get_row_contours(image)
```

## Fitting Co regions into an ellipse

The task of fitting a figure into an ellipse is called minimal volume enclosing ellipsoid. The algorithm for solving this problem was developed by [L.N. Khachiyan](https://ru.wikipedia.org/wiki/Метод_эллипсоидов). We took the implementation of this algorithm from the [Radio_Beam](https://radio-beam.readthedocs.io/en/latest/api/radio_beam.commonbeam.getMinVolEllipse.html#radio_beam.commonbeam.getMinVolEllipse) library. Ellipsoid parameters are selected so that the given points are inside the figure, and its geometric characteristics are minimal.

![Enclosed Ellipse](https://pobedit.s3.us-east-2.amazonaws.com/docs_images/enclosed-ellipse.png)

## grainMark class

### mark_corners_and_classes(image, max_num=100000, sens=0.1, max_dist=1)

**(Deprecated)** Returns all possible corner coordinates and the original image with gradient cluster classes applied

**Parameters:**
- `image`: ndarray (width, height, channels)
- `max_num`: int
- `sens`: float
- `max_dist`: int

**Returns:** corners, classes, num

### mean_pixel(image, point1, point2, r)

**(Deprecated)** Returns the mean value of pixels in a rectangle of width 2r, constructed between two points

**Parameters:**
- `image`: ndarray (width, height, channels)
- `point1`: tuple (int, int)
- `point2`: tuple (int, int)
- `r`: int

**Returns:** mean, dist

### get_row_contours(image)

Returns coordinates of contour pixels for each region

**Parameters:**
- `image`: ndarray (width, height,3)

**Returns:** list (N_contours, (M_points,2))

### get_contours(cls, image, tol=3)

Reduces the number of contour points using the Douglas-Peucker algorithm

**Parameters:**
- `image`: ndarray (width, height,3)
- `tol`: int Maximum distance from original points of polygon to approximated polygonal chain

**Returns:** list (N_contours, (M_points,2))

### get_angles(image, thr=5)

Returns angles with contour traversal direction counterclockwise, angles >180 degrees are taken into account. Accepts only preprocessed image as input

**Parameters:**
- `image`: ndarray (width, height,1), only preprocessed image
- `thr`: int, distance from original image edge to inner image edge (rect in rect)

**Returns:** angles ndarray (n), angles coords list (n_angles, 2)

### get_mvee_params(image, tol=0.2, debug=False)

Returns semi-axes and rotation angle of the minimal volume enclosing ellipsoid figure that bounds the original contour points with an ellipse. For calculations, the coordinate axis center is shifted to the polygon centroid (investigated region), and then shifted to the mean value of polygon coordinates

**Parameters:**
- `image`: ndarray (width, height,1), only preprocessed image
- `tol`: float, coefficient of ellipse compactness

**Returns:** ndarray a_beams, b_beams, angles, centroids

### gskeletons_coords(image)

Takes a binarized image as input, creates an array of individual skeleton pixels, each skeleton pixel is given a class at the coordinates where it is located. Class coordinates are determined by ndi.label

**Parameters:**
- `image`: ndarray (width, height,1)

**Returns:** bones
