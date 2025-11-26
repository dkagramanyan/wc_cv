---
title: "Morphology"
weight: 10
---

## kmeans_image(image, n_clusters)

Clusters using kmeans and returns an image with cluster colors applied

**Parameters:**
- `image`: ndarray (width, height, channels)
- `n_clusters`: int

**Returns:** (image, colors), colors - list of median colors of the clusters
