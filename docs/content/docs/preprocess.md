---
title: "Предобработка изображений"
weight: 9
---

Описанные инструменты разработаны для обработки SEM снимков.

Для явного выделения границ фаз WC/Co используется последовательное применение следующиих алгоритом

1) выбирается сторона снимка. По умолчанию использвуется снимок в отраженных электронах

2) изображение сглаживается медианным фильтром для подавления шумов и выравнивания цветового распределения пикселей

3) слаженное изображение бинаризуется при помощи метода Отсу

4) от бинаризованного изображения берется градиент. Он явно показывает переходы вас WC/Co

5) бинаризованное изобржение инвертируется и к нему добавляется карта градиентов, полученная в п.4

Полная обработка изображения выглядит следующим образом:

```
preproc_image=1-Otsu(median_filter(image))+grad(Otsu(median_filter(image)))
```

Значения пикселей по классам:

* 0 - зерно WC

* 127 - регион Co

* 254 - граница региона Co, смежного с зернами WC. Толщина границы - 1 пиксель

## Обработка одного снимка

```python
img=io.imread(img_path)
img=grainPreprocess.image_preprocess(img,h,k)
```

## Обработка всего датасета снимков

```python
all_images=grainPreprocess.read_preprocess_data(images_folder_path, images_num_per_class=150,  preprocess=True, save=True)
```

## Расположение снимков

Расположение исходных снимков и предобработанных снимков должно выглядеть следующим образом

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

## Класс grainPreprocess

### imdivide(image, h, side)

Разделяет входное изображение по середине и возвращает левую или правую часть

**Параметры:**
- `image`: ndarray (height,width,channels)
- `h`: int scalar
- `side`: float scalar

**Возвращает:** ndarray (height,width/2,channels)

### combine(image, h, k=0.5)

Накладывает левую и правые части изображения. Если k=1, то на выходе будет левая часть изображения, если k=0, то будет правая часть

**Параметры:**
- `image`: ndarray (height,width,channels)
- `h`: int scalar
- `k`: float scalar

**Возвращает:** ndarray (height,width/2,channels)

### do_otsu(image)

Бинаризация Отсу

**Параметры:**
- `img`: ndarray (height,width,channels)

**Возвращает:** ndarray (height,width), Boolean

### image_preprocess(image)

Комбинация медианного фильтра, биноризации и градиента. У зерен значение пикселя - 0, у регионов связ. в-ва - 127,а у их границы - 254. Обраотанное изображение получается следующим образом: 1-Otsu(median_filter(image))+grad(Otsu(median_filter(image)))

**Параметры:**
- `img`: ndarray (height,width,channels)

**Возвращает:** ndarray (height,width,1)

### read_preprocess_data(images_dir, max_images_num_per_class=100, preprocess=False, save=False, crop_bottom=False, h=135, resize=True, resize_shape=None, save_name='all_images.npy')

Считывание всего датасета, обработка и сохрание в .npy файл

**Параметры:**
- `images_dir`: str
- `max_images_num_per_class`: int
- `preprocess`: Bool
- `save`: Bool
- `crop_bottom`: Bool
- `h`: int
- `resize`: Bool
- `resize_shape`: tuple (width, height, channels)
- `save_name`: str

**Возвращает:** ndarray (n_classes, n_images_per_class, width, height, channels)

### tiff2jpg(folder_path, start_name=0, stop_name=-4, new_folder_path='resized')

Переводит из tiff 2^16 в jpg 2^8 бит

**Параметры:**
- `folder_path`: str
- `start_name`: int
- `stop_name`: int
- `new_folder_path`: str

**Возвращает:** None

### get_example_images()

Скачивает из контейнера s3 по 1 снимку каждого образца

**Возвращает:** ndarray [[img1],[img2]..]

### image_preprocess_kmeans(image, h=135, k=1, n_clusters=3, pos=1)

Выделение границ при помощи кластеризации и выравнивание шума медианным фильтром. Подходит только для смазанных фотограий, где границы у объектов достатчно широкие. Pos отвечает за выбор кластера, который будет отображен на возвращенном изображении

**Параметры:**
- `image`: array (height,width,channels)
- `h`: int scalar
- `k`: float scalar
- `n_clusters`: int scalar
- `pos`: int scalar, cluster index

**Возвращает:** ndarray (height,width)

