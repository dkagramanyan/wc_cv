---
title: "Ellipses and their distributions"
weight: 4
---

### Quick start

```python
from combra import mvee, data
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

Beams length distributions

```python

angles.angles_plot_base(data, plot_file_name=json_save_name, step=step, N=10, M=10, indices=[2,0,1], save=False)
```

Rotation angle distribution

```python
N = 10
M = 7

step = 2

saved_image_name='test'

mvee.plot_angles(generated_data,saved_image_name, step, N, M, indices=[3,0,2],save=False)
```

Heatmap

```python
step = 5

names_new={'Ultra_Co11':'AB_Co11_medium',
           'Ultra_Co15':'AB_Co15_medium_small',
           'Ultra_Co25':'AB_Co25_small',
           'Ultra_Co6_2':'AB_Co6_large',
           'Ultra_Co8':'AB_Co_8_medium_small'}


mvee.beams_heatmap(generated_data,saved_names=names_new,step=step, indices=[3,0,2], N=7, M=7,save=False, font_size=45, scatter_size=110)
```

Comparison

```python
file_name_1 = 'test_step_5_beams.json'

data_1 = open(file_name_1,encoding='utf-8')
data_1 = json.load(data_1)

# !!!!!
# create corresponding files with different steps mvee.diametr_approx_save
# !!!!!
file_name_2_1='test_step_2_beams.json'
file_name_2_2='test_step_3_beams.json'
file_name_2_3='test_step_4_beams.json'

save_name = 'test.png'

names = [file_name_2_1, file_name_2_2, file_name_2_3]

N = 25
M = 5


columns = f' k,% b,%'
metrics_all=[]

for file_name_2 in names:

    save_name=file_name_2.split('/')[-1]

    indices_1=[3,0, 2]
    indices_2=[2, 0, 1]
    print(columns)
    data_2 = open(file_name_2,encoding='utf-8')
    data_2 = json.load(data_2)

    metrics = mvee.mvee_plot_compare(data_1, data_2, save_name, ['b_beams'], N, M, indices_1=indices_1, indices_2=indices_2, save=False)
    metrics_all.extend(metrics)
    
print(columns)
for m in metrics_all:
    print(m)
    
```