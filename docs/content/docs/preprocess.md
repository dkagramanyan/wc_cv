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
from combra import data, image, contours

image=img2

cnt_raw=grainMark.get_row_contours(image=image)
cnt=grainMark.get_contours(image=image)

cnt_img=draw_edges(color.gray2rgb(image),cnts=cnt, r=2, e_width=1, l_width=1)

plt.figure(figsize=(5,5))
plt.imshow(cnt_img[:256,:256])
plt.show()

io.imread(image,'test.pnd')

angles = grainMark.get_angles(image,tol=3)

plt.hist(angles,bins=20)
plt.show()
```
