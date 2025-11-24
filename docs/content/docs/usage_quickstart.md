---
title: "Быстрый старт"
weight: 1
---

Ниже приведены минимальные примеры использования основных функций пакета `combra`.

## Установка

```bash
pip install -e .
```

## Предобработка одного снимка

```python
from combra import image
from skimage import io

image_raw = io.imread('path/to/image.png')
processed = image.image_preprocess(image_raw)
```

Функция `image_preprocess` выполняет:
- Медианную фильтрацию
- Бинаризацию методом Otsu
- Вычисление градиента

## Извлечение контуров

```python
from combra import contours

cnts = contours.get_contours(processed, tol=3)
print(f"Найдено контуров: {len(cnts)}")
```

## Вычисление углов

```python
from combra import angles

angles_array, contours = angles.get_angles(processed, border_eps=5, tol=3)
print(f"Найдено углов: {len(angles_array)}")
print(f"Средний угол: {angles_array.mean():.2f}°")
```

## Работа с датасетом

```python
from combra.data.dataset import SEMDataset

# Создание датасета (автоматически кэширует предобработанные изображения в /tmp)
dataset = SEMDataset('data/wc_co', max_images_num_per_class=50)

# Получение изображения и пути
image, path = dataset.__getitem__(0, 0)
print(f"Загружено изображение: {path}")

# Количество классов
print(f"Классов в датасете: {len(dataset)}")
```

## Граф трещин

### Создание графа

```python
from combra import graph, image
from skimage import io

# Предобработка изображения
img = io.imread('crack_image.png')
processed = image.image_preprocess(img)

# Предобработка для графа
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(processed, tol=5)
)

# Создание графа
G, img_contours_mono = graph.create_crack_graph(
    img_contours.shape,
    cnts,
    nodes_meta
)

print(f"Узлов в графе: {G.number_of_nodes()}")
print(f"Ребер в графе: {G.number_of_edges()}")
```

### Визуализация графа

```python
from combra import graph

graph.graph_plot(
    G,
    img_contours,
    N=10,
    M=10,
    name='crack_graph.png',
    save=True
)
```

## Визуализация контуров

```python
from combra import contours
from PIL import Image
import numpy as np

# Создание RGB изображения
img_rgb = Image.fromarray(np.zeros_like(processed, dtype=np.uint8).repeat(3, axis=2))

# Рисование контуров
img_with_contours = contours.draw_contours(
    img_rgb,
    cnts=cnts,
    color_corner=(0, 139, 139),
    color_line=(255, 140, 0),
    corners=True
)
```

## Обработка всего датасета углов

```python
from combra import angles, data
import json

# Путь к датасету
images_path = data.example_class_path()

# Словарь типов зерен
types_dict = {
    'Ultra_Co11': 'средние зерна',
    'Ultra_Co25': 'мелкие зерна',
    'Ultra_Co8': 'средне-мелкие зерна',
    'Ultra_Co6_2': 'крупные зерна',
    'Ultra_Co15': 'средне-мелкие зерна'
}

# Вычисление и сохранение распределений углов
angles.angles_approx_save(
    images_path=images_path,
    save_path='angles_results',
    types_dict=types_dict,
    step=5,  # Шаг в градусах
    max_images_num_per_class=360,
    workers=20
)

# Загрузка результатов
with open('angles_results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Визуализация
angles.angles_plot_base(
    results,
    plot_file_name='angles_plot',
    step=5,
    N=10,
    M=10,
    indices=[0, 1, 2],  # Индексы классов для отображения
    save=True
)
```

## Статистический анализ и аппроксимация

```python
from combra import stats, approx
import matplotlib.pyplot as plt
import numpy as np

# Предобработка данных
x, y = stats.stats_preprocess(angles_array, step=5)

# Бимодальная гауссова аппроксимация
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Данные', markersize=4)
plt.plot(x_gauss, y_gauss, '-', label='Аппроксимация', linewidth=2)
plt.xlabel('Угол (градусы)')
plt.ylabel('Частота')
plt.title('Распределение углов')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Мода 1: μ={mus[0]:.2f}°, σ={sigmas[0]:.2f}°")
print(f"Мода 2: μ={mus[1]:.2f}°, σ={sigmas[1]:.2f}°")
```

## MVEE (Минимальный объемлющий эллипсоид)

```python
from combra import mvee, data

images_path = data.example_class_path()
types_dict = {
    'Ultra_Co11': 'средние зерна',
    'Ultra_Co25': 'мелкие зерна'
}

# Вычисление MVEE для датасета
mvee.diametr_approx_save(
    images_path=images_path,
    save_path='mvee_results',
    types_dict=types_dict,
    step=4,
    pixel=50/1000,  # Размер пикселя в мм
    max_images_num_per_class=None
)

# Загрузка и визуализация
import json
with open('mvee_results_step_4_beams.json', 'r') as f:
    mvee_data = json.load(f)

mvee.plot_beam_base(
    mvee_data,
    save_name='mvee_plot',
    step=4,
    N=7,
    M=7,
    save=True
)
```

## Следующие шаги

Для более подробной информации см.:
- [Датасет: SEMDataset](/docs/usage_dataset) - подробнее о работе с датасетами
- [Граф трещин](/docs/usage_graph) - детали построения графов
- [Углы](/docs/usage_angles) - анализ углов и распределений
- [API Reference](/docs/api) - полный справочник API
