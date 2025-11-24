---
title: "Mark"
weight: 12
---

## Определение углов у регионов Co

Для определения углов последовательно используется 3 инстурмента:

1) поиск границ Кенни. Он находит на изображении все возможные границы и затем выделяет их

2) метод поиска конутров Suzuki. Он нужен для разметки (нумерования) пикселей найденных в п.1 контуров. Направление обхода контура - против часовой стрелки. Проверим направление. Точка - начало отрисовки контура

![Suzuki](https://pobedit.s3.us-east-2.amazonaws.com/docs_images/suzuki.jpg)

3) полученные масивы пикселей контуров аппроксимируются методом Дугласа-Пекера. Он из массива точек оставляет только те точки, которые описывают характер полигона, например: точки в углах, перегибах и тд

В итоге для каждого контура мы имеем массив точек, которые с заданной точностью описывают периметр контура.

Для определения границы воспользуемся определителем тройкой векторов. Если определитель det меньше 0, то угол phi меньше 180 градусов, иначе больше 180.

```
grainMark.get_row_contours(image)
```

## Вписывание регионов Co в эллипс

Задача вписывания фигуры в эллипс называется minimal volume enclosing ellipsoid. Алгоритм решения задачи был разработан [Л.Н.Хачуяном](https://ru.wikipedia.org/wiki/Метод_эллипсоидов). Мы взяли релизацию этого алгоритма из библиотеки [Radio_Beam](https://radio-beam.readthedocs.io/en/latest/api/radio_beam.commonbeam.getMinVolEllipse.html#radio_beam.commonbeam.getMinVolEllipse). Параметры эллипсоида подбираются так, чтобы заданные точки находились внутри фигуры, а ее геометрические характеристики были наименьшими.

![Enclosed Ellipse](https://pobedit.s3.us-east-2.amazonaws.com/docs_images/enclosed-ellipse.png)

## Класс grainMark

### mark_corners_and_classes(image, max_num=100000, sens=0.1, max_dist=1)

**(Deprecated)** Возвращает всевозможные координаты углов и исходное изображение с нанесенными классами кластеров градиента

**Параметры:**
- `image`: ndarray (width, height, channels)
- `max_num`: int
- `sens`: float
- `max_dist`: int

**Возвращает:** corners, classes, num

### mean_pixel(image, point1, point2, r)

**(Deprecated)** Возвращает среднее значение пикселей прямоугольника ширины 2r, построеного между двумя точками

**Параметры:**
- `image`: ndarray (width, height, channels)
- `point1`: tuple (int, int)
- `point2`: tuple (int, int)
- `r`: int

**Возвращает:** mean, dist

### get_row_contours(image)

Возвращает кооридинаты пикселей контуров каждого региона

**Параметры:**
- `image`: ndarray (width, height,3)

**Возвращает:** list (N_contours, (M_points,2))

### get_contours(cls, image, tol=3)

Уменьшение количества точек контура при помощи алгоритма Дугласа-Пекера

**Параметры:**
- `image`: ndarray (width, height,3)
- `tol`: int Maximum distance from original points of polygon to approximated polygonal chain

**Возвращает:** list (N_contours, (M_points,2))

### get_angles(image, thr=5)

Возвращает углы с направлением обхода контура против часовой стрелки, углы >180 градусов учитываются. На вход принимает только обработанное изображение

**Параметры:**
- `image`: ndarray (width, height,1), only preprocessed image
- `thr`: int, distance from original image edge to inner image edge (rect in rect)

**Возвращает:** angles ndarray (n), angles coords list (n_angles, 2)

### get_mvee_params(image, tol=0.2, debug=False)

Возвращает полуоси и угол поворота фигуры minimal volume enclosing ellipsoid, которая ограничивает исходные точки контура эллипсом. Для расчетов центр координатной оси сдвигается на центроид полигона (исследуемого региона), а затем сдвигается на среднее значение координат полигона

**Параметры:**
- `image`: ndarray (width, height,1), only preprocessed image
- `tol`: foat, koef of ellipse compactness

**Возвращает:** ndarray a_beams, b_beams, angles, centroids

### gskeletons_coords(image)

На вход подается бинаризованное изображение создает массив индивидуальных скелетов пикселю скелета дается класс, на координатах которого он находится. Координаты класса определяются ndi.label

**Параметры:**
- `image`: ndarray (width, height,1)

**Возвращает:** bones

