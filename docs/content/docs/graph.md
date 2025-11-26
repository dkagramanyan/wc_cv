---
title: "Crack Graph"
weight: 5
---



### Quickstart

**Unlabeled data**

Generate data

```python
from combra import graph, data
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd

image = data.example_crack_fixed_images()[0][1]

(entry_nodes,
 exit_nodes,
 img_contours_o,
 img_preprocessed_final,
 cnts,
 nodes_metadata)  = graph.preprocess_graph_image(image, border=30, disk=5,entry_ellps_w=5,exit_ellps_w=5,r=4)

g, img_contours =  graph.create_crack_graph(img_preprocessed_final.shape, cnts, nodes_metadata, eps=300)

```

Plot graph

```python

graph.graph_plot(g, img_preprocessed_final,N=10,M=10, name='wc_cv/cv/Ultra_Co6_2-001_cut_graph.jpeg', save=False)

```

Plot paths (doesnt work)

```python

## DOESN'T WORK

entry_nodes = np.unique(paths['entry_node'])
exit_nodes = np.unique(paths['exit_node'])

shortest_entry_paths = []
for entry_node in tqdm(entry_nodes):
    row = paths[paths.entry_node==entry_node].sort_values(by='path_len_pixels').iloc[0]
    shortest_entry_paths.append(row)

shortest_exit_paths = []
for exit_node in tqdm(exit_nodes):
    row = paths[paths.exit_node==exit_node].sort_values(by='path_len_pixels').iloc[0]
    shortest_exit_paths.append(row)

df_shortest_entry = pd.DataFrame(shortest_entry_paths)
df_shortest_exit = pd.DataFrame(shortest_exit_paths)


graph.plot_paths(g, df, img_aligned, border=30)

```

Generate energies

```python
entry_nodes = [ 0,1, 3, 9, 10, 13]
exit_nodes = [ 23, 24, 71, 72, 63, 64, 56, 57, 67, 68]

# WC+8Co_5_fixed_001_cropped.jpg
# entry_nodes = [ 4, 15, 13, 16, 34, 27]
# exit_nodes = [ 527, 528, 519, 522]

# WC+8Co_5_crack
# entry_nodes = [ 4, 11, 16, 61, 62, 66]
# exit_nodes = [ 307, 308, 515, 529]

param_1_max=20 # Co
param_2_max=20 # WC-Co
param_3_const=20 # WC
param_4_const=0 # WC-WC

energy_conf=np.zeros((param_1_max, param_2_max)).tolist()

for i,en_1 in enumerate(tqdm(range(0,param_1_max))):
    for j,en_2 in enumerate(range(0,param_2_max)):

        energy_conf[i][j]={
            0: en_1, # Co
            1: en_2, # WC-Co
            2: param_3_const, # WC
            3: param_4_const # WC-WC
            } 

energies_paths = graph.get_energies(energy_conf,
                                    g,
                                    cnts,
                                    nodes_metadata,
                                    entry_nodes,
                                    exit_nodes,
                                    first_k_paths=1,
                                    parallel=True,
                                    workers=20)
```

Plot energy optimized paths (doesnt work)

```python 
# does not work well with large images

graph.plot_optimized_paths(g, energies_paths, img_contours_o)
```

Plot energies

```python
path_index=0

graph.plot_optimized_energies(  energies_paths,
                                path_index=path_index,
                                N=5,M=5,
                                y_label ='co_e',
                                x_label = 'wc-co_e',
                                fontsize_h=10,
                                fontsize_axes=50
                                )
```

Plot fixed path energy
```python
# WC+8Co_5_crack
# entry_nodes = [ 4, 11, 16, 61, 62, 66]
# exit_nodes = [ 307, 308, 515, 529]

# WC+8Co_5_fixed_001_cropped.jpg
entry_nodes = [ 4, 15, 13, 16, 34, 27]
exit_nodes = [ 527, 528, 519, 522]

fixed_paths_energies = graph.fixed_paths_energies(g, cnts, nodes_metadata, entry_nodes, exit_nodes)
```

