---
title: "Углы и их распределения"
weight: 4
---

# Углы и их распределения

Модуль `angles` предоставляет функциональность для вычисления углов на контурах зерен и анализа их распределений.

## Обзор

Углы вычисляются на упрощенных контурах зерен:
- Контуры упрощаются алгоритмом Дугласа-Пекера
- Для каждой точки контура вычисляется угол между соседними сегментами
- Углы измеряются в градусах (0-360°)
- Учитывается направление обхода контура (против часовой стрелки)

## Вычисление углов для одного изображения

### `get_angles()`

Основная функция для вычисления углов на предобработанном изображении.

```python
from combra import angles, image
from skimage import io

# Предобработка изображения
img = io.imread('image.png')
processed = image.image_preprocess(img)

# Вычисление углов
angles_array, contours = angles.get_angles(
    processed,
    border_eps=5,  # Расстояние от края изображения
    tol=3          # Точность упрощения контуров
)

print(f"Найдено углов: {len(angles_array)}")
print(f"Средний угол: {angles_array.mean():.2f}°")
print(f"Медианный угол: {np.median(angles_array):.2f}°")
```

**Параметры:**
- `image` (ndarray): Предобработанное изображение (width, height, 1)
- `border_eps` (int): Расстояние от края изображения до внутреннего края (игнорируются контуры слишком близко к краю)
- `tol` (int): Точность упрощения контуров алгоритмом Дугласа-Пекера

**Возвращает:**
- `angles` (ndarray): Массив углов в градусах
- `contours` (list): Список обработанных контуров

**Пример использования:**
```python
import numpy as np
import matplotlib.pyplot as plt
from combra import angles, image, stats, approx
from skimage import io

# Загрузка и предобработка
img = io.imread('grain_image.png')
processed = image.image_preprocess(img)

# Вычисление углов
angles_array, cnts = angles.get_angles(processed, border_eps=5, tol=3)

# Статистика
print(f"Всего углов: {len(angles_array)}")
print(f"Минимальный угол: {angles_array.min():.2f}°")
print(f"Максимальный угол: {angles_array.max():.2f}°")
print(f"Средний угол: {angles_array.mean():.2f}°")
print(f"Стандартное отклонение: {angles_array.std():.2f}°")

# Гистограмма
plt.hist(angles_array, bins=36, edgecolor='black')
plt.xlabel('Угол (градусы)')
plt.ylabel('Частота')
plt.title('Распределение углов')
plt.show()
```

## Обработка всего датасета

### `angles_approx_save()`

Вычисляет и сохраняет распределения углов для всех изображений в датасете с автоматической аппроксимацией.

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

# Вычисление и сохранение
angles.angles_approx_save(
    images_path=images_path,
    save_path='angles_results',
    types_dict=types_dict,
    step=5,                      # Шаг угла в градусах
    max_images_num_per_class=360,
    workers=20                   # Количество процессов
)
```

**Параметры:**
- `images_path` (str): Путь к папке с классами изображений
- `save_path` (str): Базовое имя для сохранения JSON файла
- `types_dict` (dict): Словарь соответствия имен классов и их описаний
- `step` (int): Шаг угла в градусах для гистограммы
- `max_images_num_per_class` (int, optional): Максимум изображений на класс
- `workers` (int, optional): Количество процессов для параллельной обработки

**Результат:**
Создается JSON файл с именем `{save_path}_step_{step}_degrees.json` содержащий:
- Пути к изображениям
- Имена классов
- Текстовые легенды с параметрами распределений
- Данные для визуализации (гистограммы)
- Параметры гауссовой аппроксимации

**Структура JSON файла:**
```json
[
  {
    "path": "/path/to/class",
    "name": "Ultra_Co11",
    "type": "средние зерна",
    "legend": "--------------\nUltra_Co11 средние зерна\n количество углов 1000\n...",
    "density_curve_scatter": [[0, 5, 10, ...], [10, 15, 20, ...]],
    "gauss_approx_plot": [[0, 1, 2, ...], [5.2, 5.5, 5.8, ...]],
    "gauss_approx_data": {
      "mus": [90.5, 270.3],
      "sigmas": [30.2, 28.7],
      "amps": [1.0, 0.95]
    }
  },
  ...
]
```

## Загрузка и анализ результатов

```python
import json

# Загрузка результатов
with open('angles_results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Анализ каждого класса
for item in data:
    print(f"\nКласс: {item['name']}")
    print(f"Тип: {item['type']}")
    print(f"Легенда:\n{item['legend']}")
    
    # Данные для визуализации
    x, y = item['density_curve_scatter']
    print(f"Точек в распределении: {len(x)}")
    
    # Параметры аппроксимации
    mus = item['gauss_approx_data']['mus']
    sigmas = item['gauss_approx_data']['sigmas']
    amps = item['gauss_approx_data']['amps']
    
    print(f"Мода 1: μ={mus[0]:.2f}°, σ={sigmas[0]:.2f}°")
    print(f"Мода 2: μ={mus[1]:.2f}°, σ={sigmas[1]:.2f}°")
```

## Визуализация

### `angles_plot_base()`

Визуализация распределений углов для нескольких классов.

```python
from combra import angles
import json

# Загрузка данных
with open('angles_results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Визуализация
angles.angles_plot_base(
    data,
    plot_file_name='angles_plot',
    step=5,
    N=10,              # Количество строк в сетке
    M=10,              # Количество столбцов в сетке
    indices=[0, 1, 2], # Индексы классов для отображения
    save=True,
    font_size=20,
    scatter_size=20
)
```

**Параметры:**
- `data`: Данные из JSON файла
- `plot_file_name`: Базовое имя для сохранения графиков
- `step`: Шаг угла в градусах
- `N`, `M`: Размеры сетки графиков
- `indices`: Список индексов классов для отображения (если `None`, отображаются все)
- `save`: Сохранить графики
- `font_size`: Размер шрифта
- `scatter_size`: Размер точек на графике

### `angles_approx_modes()`

Визуализация мод распределения углов.

```python
from combra import angles

angles.angles_approx_modes(
    folder='results',      # Папка с JSON файлами
    step=5,
    start1=0, stop1=180,   # Диапазон первой моды
    start2=180, stop2=360, # Диапазон второй моды
    width=10,
    height=10,
    font_size=25
)
```

## Статистический анализ

### Интеграция с модулем stats

```python
from combra import angles, stats, approx
import matplotlib.pyplot as plt
import numpy as np

# Вычисление углов
angles_array, _ = angles.get_angles(processed, border_eps=5, tol=3)

# Предобработка для статистики
x, y = stats.stats_preprocess(angles_array, step=5)

# Бимодальная гауссова аппроксимация
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'o', label='Данные', markersize=4, alpha=0.6)
plt.plot(x_gauss, y_gauss, '-', label='Бимодальная аппроксимация', linewidth=2)
plt.xlabel('Угол (градусы)', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.title('Распределение углов с аппроксимацией', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

print(f"Мода 1: μ={mus[0]:.2f}°, σ={sigmas[0]:.2f}°, A={amps[0]:.2f}")
print(f"Мода 2: μ={mus[1]:.2f}°, σ={sigmas[1]:.2f}°, A={amps[1]:.2f}")
```

## Создание легенды

### `angles_legend()`

Создание текстовой легенды для распределения углов.

```python
from combra import angles

legend = angles.angles_legend(
    images_amount=100,      # Количество изображений
    name='Ultra_Co11',      # Имя класса
    itype='средние зерна',  # Тип зерен
    step=5,                 # Шаг угла
    mus=[90.5, 270.3],      # Средние значения мод
    sigmas=[30.2, 28.7],    # Стандартные отклонения
    amps=[1.0, 0.95],       # Амплитуды
    norm=1000               # Общее количество углов
)

print(legend)
```

## Полный пример

```python
from combra import angles, image, data, stats, approx
from skimage import io
import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Обработка одного изображения
img = io.imread('grain_image.png')
processed = image.image_preprocess(img)
angles_array, cnts = angles.get_angles(processed, border_eps=5, tol=3)

# 2. Статистический анализ
x, y = stats.stats_preprocess(angles_array, step=5)
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# 3. Визуализация
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'o', label='Данные', markersize=4)
plt.plot(x_gauss, y_gauss, '-', label='Аппроксимация', linewidth=2)
plt.xlabel('Угол (градусы)')
plt.ylabel('Частота')
plt.legend()
plt.show()

# 4. Обработка всего датасета
images_path = data.example_class_path()
types_dict = {
    'Ultra_Co11': 'средние зерна',
    'Ultra_Co25': 'мелкие зерна'
}

angles.angles_approx_save(
    images_path=images_path,
    save_path='results',
    types_dict=types_dict,
    step=5,
    max_images_num_per_class=100
)

# 5. Загрузка и визуализация результатов
with open('results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

angles.angles_plot_base(
    results,
    plot_file_name='angles_plot',
    step=5,
    N=10,
    M=10,
    save=True
)
```

## Рекомендации по параметрам

### `border_eps`
- **Маленькие значения (3-5)**: Больше углов, но могут быть артефакты на краях
- **Большие значения (10-15)**: Меньше углов, но более надежные результаты

### `tol` (точность упрощения)
- **Маленькие значения (1-3)**: Более точные контуры, больше точек, медленнее
- **Большие значения (5-10)**: Меньше точек, быстрее, но может теряться детализация

### `step` (шаг для гистограммы)
- **5 градусов**: Стандартное значение, хороший баланс
- **1-2 градуса**: Высокая детализация, больше шума
- **10 градусов**: Меньше деталей, более гладкие распределения

## Примечания

- Углы вычисляются с учетом направления обхода контура (против часовой стрелки)
- Углы >180° учитываются (полный диапазон 0-360°)
- Контуры слишком близко к краю изображения игнорируются
- Для больших датасетов рекомендуется использовать параллельную обработку (`workers` параметр)
