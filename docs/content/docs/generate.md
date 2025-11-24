---
title: "Generate"
weight: 17
---

## angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm)

Cоздание легенды распределения углов

**Параметры:**
- `images_amount`: int
- `name`: str
- `itype`: str
- `step`: int
- `mus`: float
- `sigmas`: float
- `amps`: float
- `norm`: int

**Возвращает:** str

## angles_approx_save(folder, images, names, types, step, save=True)

Вычисление и сохранение распределения углов для всех фотографий одного образца

**Параметры:**
- `folder`: str path to dir
- `images`: ndarray uint8 [[image1_class1,image2_class1,..],[image1_class2,image2_class2,..]..]
- `names`: list str [class_name1,class_name2,..]
- `types`: list str [class_type1,class_type2,..]
- `step`: scalar int [0,N]
- `save`: bool

## beams_legend(name, itype, norm, k, angle, b, score, dist_step, dist_mean)

Создание легенды для распределения длин полуосей

**Параметры:**
- `name`: str
- `itype`: str
- `norm`: int
- `k`: float
- `angle`: float
- `b`: float
- `score`: float
- `dist_step`: int
- `dist_mean`: float

**Возвращает:** str

## diametr_approx_save(folder, images, names, types, step, pixel, start=2, end=-3, save=True, debug=False)

Вычисление и сохранение распределения длин а- и б- полуосей и угла поворота эллипса для разных образцов

**Параметры:**
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

**Возвращает:** None

