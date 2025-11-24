---
title: "Morphology"
weight: 10
---

## kmeans_image(image, n_clusters)

Кластеризует при помощи kmeans и возвращает изображение с нанесенными цветами кластеров

**Параметры:**
- `image`: ndarray (width, height, channels)
- `n_clusters`: int

**Возвращает:** (image, colors), colors - list of median colors of the clusters

