---
title: "Get started"
weight: 0
---


##  Usage

```
from combra import data, angles
from pathlib import Path
import pyarrow.parquet as pq

images_folder_path=data.microstructure_class_path()

dataset = data.PobeditDataset(images_folder_path=images_folder_path,
                              max_images_num_per_class=None)


types_dict = {'Ultra_Co11': 'средние зерна',
              'Ultra_Co25': 'мелкие зерна',
              'Ultra_Co8': 'средне-мелкие зерна',
              'Ultra_Co6_2': 'крупные зерна',
              'Ultra_Co15': 'средне-мелкие зерна'}

pixel = 50/1000

for step in [5]:
    dataset.generate(f'orig_step={step}.parquet', types_dict=types_dict, step=step, workers=20, pixel = pixel, angles_tol=3,mvee_tol=0.2, start=0,end=-1,)


in_path = Path("orig_step=5.parquet")

table = pq.read_table(in_path)
rows = table.to_pydict()

angles.angles_plot_base(rows,save_name="orig_step=5.parquet",N=15,M=7,  save=False,indices=None, font_size=20,scatter_size=20)
```