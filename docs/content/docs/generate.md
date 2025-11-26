---
title: "Generate"
weight: 17
---

## angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm)

Creates a legend for angle distribution

**Parameters:**
- `images_amount`: int
- `name`: str
- `itype`: str
- `step`: int
- `mus`: float
- `sigmas`: float
- `amps`: float
- `norm`: int

**Returns:** str

## angles_approx_save(folder, images, names, types, step, save=True)

Computes and saves angle distribution for all photos of one sample

**Parameters:**
- `folder`: str path to dir
- `images`: ndarray uint8 [[image1_class1,image2_class1,..],[image1_class2,image2_class2,..]..]
- `names`: list str [class_name1,class_name2,..]
- `types`: list str [class_type1,class_type2,..]
- `step`: scalar int [0,N]
- `save`: bool

## beams_legend(name, itype, norm, k, angle, b, score, dist_step, dist_mean)

Creates a legend for semi-axis length distribution

**Parameters:**
- `name`: str
- `itype`: str
- `norm`: int
- `k`: float
- `angle`: float
- `b`: float
- `score`: float
- `dist_step`: int
- `dist_mean`: float

**Returns:** str

## diametr_approx_save(folder, images, names, types, step, pixel, start=2, end=-3, save=True, debug=False)

Computes and saves distribution of a- and b- semi-axis lengths and ellipse rotation angle for different samples

**Parameters:**
- `folder`: str
- `images`: ndarray uint8 [[image1_class1,image2_class1,..],[image1_class2,image2_class2,..]..]
- `names`: list str [class_name1,class_name2,..]
- `types`: list str [class_type1,class_type2,..]
- `step`: scalar int [0,N]
- `pixel`: float
- `start`: int
- `end`: int
- `save`: bool
- `debug`: bool

**Returns:** None
