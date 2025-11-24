---
title: "Быстрый старт"
weight: 1
---

Ниже приведены минимальные примеры использования основных функций пакета `wc_cv`.

## Предобработка одного снимка

```python
from skimage import io
from wc_cv import grainPreprocess

image = io.imread('path/to/image.png')
processed = grainPreprocess.image_preprocess(image)
```

## Извлечение контуров и углов

```python
from wc_cv import grainMark

angles, contours = grainMark.get_angles(processed)
```

## Работа с датасетом (кэширование в /tmp)

```python
from wc_cv import SEMDataset

# подготовит кэш и вернёт индексы путей
dataset = SEMDataset('dataset_root', max_images_num_per_class=50)
image, path = dataset.__getitem__(0, 0)
```

## Граф трещин (создание, визуализация)

```python
from wc_cv import Crack, grainDraw
from PIL import Image
import numpy as np

# подготовка данных (пример)
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = Crack.preprocess_graph_image(processed)

# граф
G, img = Crack.create_crack_graph(img_contours.shape, cnts, nodes_meta)

# картинка с контурами
img_rgb = grainDraw.draw_contours(Image.fromarray(np.zeros_like(img_contours)), cnts=cnts)
```

Подробнее см. страницы "Датасет", "Граф трещин" и "Углы".

