---
title: "Areas"
weight: 3
---

### Quickstart



```python
from combra import mvee, data, areas
import json


pixel = 50 / 1000
step = 4


images_path = data.example_class_path()
json_save_name = 'test'

types_dict = {'Ultra_Co11': 'средние зерна',
              'Ultra_Co25': 'мелкие зерна',
              'Ultra_Co8': 'средне-мелкие зерна',
              'Ultra_Co6_2': 'крупные зерна',
              'Ultra_Co15': 'средне-мелкие зерна'}

mvee.diametr_approx_save(
                        images_path=images_path,
                        save_path=json_save_name,
                        types_dict=types_dict,
                        step=step,
                        max_images_num_per_class=None, 
                        pixel = pixel
    )

generated_data = open(json_save_name+f'_step_{step}_beams.json', encoding='utf-8')
generated_data = json.load(generated_data)
```

Area distribution

```python
N = 15
M = 10
step = 1

saved_image_name = f'original'

areas.plot_polygons_area(generated_data, saved_image_name, step, N, M, indices=[3,0,2], save=False, start=0, log_min_val=-10,min_area_num=10)

```

Effective radius distribution

```python
N = 20
M = 20

step = 2

saved_image_name='test'

areas.plot_polygons_effective_radius(generated_data,saved_image_name, step, N, M, indices=[3,0,2],save=False)
```