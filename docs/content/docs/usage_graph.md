---
title: "Граф трещин: Crack"
weight: 3
---

## Пайплайн

1. Предобработка входного изображения (``grainPreprocess.image_preprocess``).
2. ``Crack.preprocess_graph_image`` — выделение контуров, подготовка метаданных узлов.
3. ``Crack.create_crack_graph`` — построение ориентированного графа по контурам.
4. (Опционально) ``Energy.get_energies`` — расчёт энергий путей.
5. ``Viz.graph_plot`` — визуализация.

## Минимальный пример

```python
from wc_cv import grainPreprocess, Crack, Viz
from skimage import io

image = io.imread('path/to/img.png')
proc = grainPreprocess.image_preprocess(image)

entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    Crack.preprocess_graph_image(proc, tol=5)
)

G, img_contours_mono = Crack.create_crack_graph(
    img_contours.shape, cnts, nodes_meta
)

Viz.graph_plot(G, img_contours_mono, N=10, M=10)
```

## Параметры и классы

- ``Crack.get_edge_type`` — типы рёбер: 0 Co, 1 WC-Co, 2 WC, 3 WC-WC.
- ``Crack.get_edge_type_labeled`` — определение типа на размеченных данных.
- ``Viz.plot_optimized_energies`` — матрицы энергий по конфигурациям.
- ``Energy.find_shortest_energy_paths`` — кратчайшие пути по весам.

