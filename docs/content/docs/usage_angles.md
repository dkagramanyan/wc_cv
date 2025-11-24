---
title: "Углы и их распределения"
weight: 4
---

## Извлечение углов

```python
from wc_cv import grainPreprocess, grainMark
from skimage import io

image = io.imread('path.png')
proc = grainPreprocess.image_preprocess(image)
angles, contours = grainMark.get_angles(proc, border_eps=5, tol=3)
```

```python
def hello(name):
    print(f"Hello, {name}")

hello("world")
```

## Предобработка статистики

```python
from wc_cv import grainStats

x, y = grainStats.stats_preprocess(angles, step=5)
```

## Бимодальная аппроксимация

```python
from wc_cv import grainApprox

(x_plot, y_plot), mus, sigmas, amps = grainApprox.bimodal_gauss_approx(x, y)
```

## Визуализация

```python
from wc_cv import grainShow
# см. также grainShow.angles_plot_base для комплексных графиков
```

