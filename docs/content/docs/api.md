---
title: "API Reference"
weight: 21
---

# API Reference

Полный справочник API для библиотеки `combra` - компьютерное зрение для композитных сплавов.

## Модули

Библиотека `combra` организована в следующие модули:

- [`data`](#data) - Работа с датасетами и примерами данных
- [`image`](#image) - Предобработка изображений
- [`contours`](#contours) - Извлечение и обработка контуров
- [`angles`](#angles) - Вычисление углов и их распределений
- [`graph`](#graph) - Построение и анализ графов трещин
- [`mvee`](#mvee) - Минимальный объемлющий эллипсоид (MVEE)
- [`areas`](#areas) - Анализ площадей полигонов
- [`approx`](#approx) - Аппроксимация данных
- [`stats`](#stats) - Статистическая обработка и распределения

---

## data

Модуль для работы с датасетами изображений.

### SEMDataset

Класс для загрузки и предобработки датасета изображений с автоматическим кэшированием.

```python
from combra.data.dataset import SEMDataset

dataset = SEMDataset(
    images_folder_path='path/to/images',
    max_images_num_per_class=100,
    workers=None  # По умолчанию: cpu_count()-1
)

# Получить изображение и путь
image, path = dataset.__getitem__(class_idx=0, idx=0)

# Количество классов
num_classes = len(dataset)
```

**Параметры:**
- `images_folder_path` (str): Путь к корневой папке с подпапками-классами (не должен оканчиваться на `/`)
- `max_images_num_per_class` (int): Максимальное количество изображений на класс
- `workers` (int, optional): Количество процессов для параллельной обработки

**Методы:**
- `__getitem__(class_idx, idx)`: Возвращает кортеж `(image, path)` для указанного класса и индекса
- `__len__()`: Возвращает количество классов
- `preprocess_image(image, pad=False, border=30, disk=3)`: Статический метод для предобработки одного изображения

**Пример:**
```python
from combra.data.dataset import SEMDataset

# Создание датасета (автоматически кэширует в /tmp)
dataset = SEMDataset('data/wc_co', max_images_num_per_class=50)

# Итерация по всем классам и изображениям
for class_idx in range(len(dataset)):
    for img_idx in range(dataset.images_paths.shape[1]):
        image, path = dataset.__getitem__(class_idx, img_idx)
        # Обработка изображения
        print(f"Class {class_idx}, Image {img_idx}: {path}")
```

### example_images()

Возвращает список примеров изображений из встроенного датасета.

```python
from combra import data

images = data.example_images()
# Возвращает: List[Tuple[str, np.ndarray]]
```

### example_class_path()

Возвращает путь к папке с примерами классов.

```python
from combra import data

path = data.example_class_path()
# Возвращает: str
```

### example_crack_fixed_images()

Возвращает список примеров изображений трещин.

```python
from combra import data

images = data.example_crack_fixed_images()
# Возвращает: List[Tuple[str, np.ndarray]]
```

---

## image

Модуль для предобработки изображений.

### image_preprocess(image)

Основная функция предобработки изображения: медианный фильтр, бинаризация Otsu и градиент.

```python
from combra import image
from skimage import io

img = io.imread('path/to/image.png')
processed = image.image_preprocess(img)
```

**Параметры:**
- `image` (ndarray): Входное изображение (height, width, channels)

**Возвращает:**
- `ndarray`: Предобработанное изображение (height, width, 1)

### preprocess_image(image, pad=False, border=30, disk=3)

Предобработка изображения с опциональным добавлением границы.

```python
from combra.data.dataset import SEMDataset

processed = SEMDataset.preprocess_image(
    image,
    pad=True,
    border=30,
    disk=3
)
```

**Параметры:**
- `image` (ndarray): Входное изображение
- `pad` (bool): Добавить границу
- `border` (int): Размер границы в пикселях
- `disk` (int): Размер диска для медианного фильтра

### do_otsu(img)

Бинаризация изображения методом Otsu.

```python
from combra import image

binary = image.do_otsu(img)
```

### align_figures(orig_img_padded, tol, labeled_cnts=False, labels=False)

Выравнивание фигур на изображении.

```python
from combra import image

aligned = image.align_figures(
    orig_img_padded,
    tol=5,
    labeled_cnts=False
)
```

### fill_polygon(grid, corners, fill_value=1)

Заполнение полигона на сетке.

```python
from combra import image

filled = image.fill_polygon(grid, corners, fill_value=1)
```

### get_perp_v(start_node_x, start_node_y, end_node_x, end_node_y, line_eps=10)

Вычисление перпендикулярного вектора к линии.

```python
from combra import image

perp_vector = image.get_perp_v(x1, y1, x2, y2, line_eps=10)
```

### find_intersection_2d(p1, p2, p3, p4)

Нахождение точки пересечения двух отрезков в 2D.

```python
from combra import image

intersection = image.find_intersection_2d(p1, p2, p3, p4)
```

### is_point_in_polygon(x, y, corners)

Проверка, находится ли точка внутри полигона.

```python
from combra import image

inside = image.is_point_in_polygon(x, y, corners)
```

### get_bresenham_eps_pixels(img_contours_np, start_node_x, start_node_y, end_node_x, end_node_y, border_pixel=255)

Получение пикселей вдоль линии Брезенхема с эпсилон-окрестностью.

```python
from combra import image

pixels = image.get_bresenham_eps_pixels(
    img_contours_np,
    x1, y1, x2, y2,
    border_pixel=255
)
```

---

## contours

Модуль для работы с контурами.

### get_row_contours(image)

Извлечение сырых контуров из изображения.

```python
from combra import contours

cnts = contours.get_row_contours(image)
# Возвращает: list of ndarray, каждый элемент - массив координат точек контура
```

**Параметры:**
- `image` (ndarray): Входное изображение (width, height, 3)

**Возвращает:**
- `list`: Список контуров, каждый контур - массив формы (M_points, 2)

### get_contours(image, tol=3)

Извлечение контуров с упрощением алгоритмом Дугласа-Пекера.

```python
from combra import contours

cnts = contours.get_contours(image, tol=3)
```

**Параметры:**
- `image` (ndarray): Входное изображение
- `tol` (int): Максимальное расстояние от исходных точек до упрощенной полигональной цепи

**Возвращает:**
- `list`: Список упрощенных контуров

**Пример:**
```python
from combra import image, contours
from skimage import io

# Предобработка
img = io.imread('image.png')
processed = image.image_preprocess(img)

# Извлечение контуров
cnts = contours.get_contours(processed, tol=3)
print(f"Найдено контуров: {len(cnts)}")
```

### draw_contours(image, cnts, color_corner=(0, 139, 139), color_line=(255, 140, 0), r=2, e_width=5, l_width=2, corners=False)

Визуализация контуров на изображении.

```python
from combra import contours
from PIL import Image
import numpy as np

img_rgb = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
img_with_contours = contours.draw_contours(
    img_rgb,
    cnts,
    color_corner=(0, 139, 139),
    color_line=(255, 140, 0),
    corners=True
)
```

**Параметры:**
- `image`: PIL Image или ndarray
- `cnts`: Список контуров
- `color_corner`: Цвет углов (RGB)
- `color_line`: Цвет линий (RGB)
- `r`: Радиус точек углов
- `e_width`: Ширина линий контуров
- `l_width`: Ширина линий
- `corners`: Показывать углы

### draw_edges(image, cnts, color=(0, 139, 139), r=4, e_width=5, l_width=4)

Визуализация ребер контуров.

```python
from combra import contours

img_with_edges = contours.draw_edges(image, cnts, color=(0, 139, 139))
```

### mark_corners_and_classes(image, max_num=100000, sens=0.1, max_dist=1)

Обнаружение углов и классификация по градиентам.

```python
from combra import contours

corners, classes, num = contours.mark_corners_and_classes(
    image,
    max_num=100000,
    sens=0.1,
    max_dist=1
)
```

### skeletons_coords(image)

Извлечение координат скелетов из бинаризованного изображения.

```python
from combra import contours

bones = contours.skeletons_coords(binary_image)
```

---

## angles

Модуль для вычисления углов и анализа их распределений.

### get_angles(image, border_eps=5, tol=3)

Вычисление углов на контурах изображения.

```python
from combra import angles
from combra import image

processed = image.image_preprocess(img)
angles_array, contours = angles.get_angles(processed, border_eps=5, tol=3)
```

**Параметры:**
- `image` (ndarray): Предобработанное изображение (width, height, 1)
- `border_eps` (int): Расстояние от края изображения до внутреннего края
- `tol` (int): Точность упрощения контуров

**Возвращает:**
- `angles` (ndarray): Массив углов в градусах
- `contours` (list): Список обработанных контуров

**Пример:**
```python
from combra import angles, image, data
from skimage import io

# Загрузка и предобработка
img = io.imread('image.png')
processed = image.image_preprocess(img)

# Вычисление углов
angles_array, cnts = angles.get_angles(processed, border_eps=5, tol=3)
print(f"Найдено углов: {len(angles_array)}")
print(f"Средний угол: {angles_array.mean():.2f}°")
```

### angles_approx_save(images_path, save_path, types_dict, step, max_images_num_per_class=None, workers=None)

Вычисление и сохранение распределения углов для всех изображений датасета.

```python
from combra import angles, data

images_path = data.example_class_path()
types_dict = {
    'Ultra_Co11': 'средние зерна',
    'Ultra_Co25': 'мелкие зерна',
    'Ultra_Co8': 'средне-мелкие зерна'
}

angles.angles_approx_save(
    images_path=images_path,
    save_path='angles_results',
    types_dict=types_dict,
    step=5,  # Шаг в градусах
    max_images_num_per_class=360,
    workers=20
)
```

**Параметры:**
- `images_path` (str): Путь к папке с классами изображений
- `save_path` (str): Базовое имя для сохранения JSON файла
- `types_dict` (dict): Словарь соответствия имен классов и их описаний
- `step` (int): Шаг угла в градусах для гистограммы
- `max_images_num_per_class` (int, optional): Максимум изображений на класс
- `workers` (int, optional): Количество процессов

**Возвращает:**
- Сохраняет JSON файл с именем `{save_path}_step_{step}_degrees.json`

**Пример использования сохраненных данных:**
```python
import json

with open('angles_results_step_5_degrees.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    print(f"Класс: {item['name']}")
    print(f"Легенда: {item['legend']}")
    x, y = item['density_curve_scatter']
    print(f"Точек в распределении: {len(x)}")
```

### angles_plot_base(data, plot_file_name, step, N, M, save=False, indices=None, font_size=20, scatter_size=20)

Визуализация распределения углов.

```python
from combra import angles
import json

with open('angles_results_step_5_degrees.json', 'r') as f:
    data = json.load(f)

angles.angles_plot_base(
    data,
    plot_file_name='angles_plot',
    step=5,
    N=10,  # Количество строк в сетке
    M=10,  # Количество столбцов в сетке
    indices=[0, 1, 2],  # Индексы классов для отображения
    save=True
)
```

### angles_approx_modes(folder, step, start1, stop1, start2, stop2, width, height, font_size=25)

Визуализация мод распределения углов.

```python
from combra import angles

angles.angles_approx_modes(
    folder='results',
    step=5,
    start1=0, stop1=180,
    start2=180, stop2=360,
    width=10, height=10,
    font_size=25
)
```

### angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm)

Создание текстовой легенды для распределения углов.

```python
from combra import angles

legend = angles.angles_legend(
    images_amount=100,
    name='Ultra_Co11',
    itype='средние зерна',
    step=5,
    mus=[90, 270],
    sigmas=[30, 30],
    amps=[1.0, 1.0],
    norm=1000
)
```

---

## graph

Модуль для построения и анализа графов трещин.

### preprocess_graph_image(image, r=2, border=30, border_node_eps=10, tol=5, disk=5, labeled_cnts=False, labels=False, ...)

Предобработка изображения для построения графа трещин.

```python
from combra import graph, image
from skimage import io

img = io.imread('crack_image.png')
processed = image.image_preprocess(img)

entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        processed,
        r=2,
        border=30,
        border_node_eps=10,
        tol=5
    )
)
```

**Возвращает:**
- `entry_nodes`: Входные узлы графа
- `exit_nodes`: Выходные узлы графа
- `img_contours`: Изображение с контурами
- `img_marked`: Размеченное изображение
- `cnts`: Список контуров
- `nodes_meta`: Метаданные узлов

### create_crack_graph(img_shape, cnts, nodes_metadata, ...)

Создание ориентированного графа трещин.

```python
from combra import graph
import networkx as nx

G, img_contours_mono = graph.create_crack_graph(
    img_contours.shape,
    cnts,
    nodes_meta
)

print(f"Узлов: {G.number_of_nodes()}")
print(f"Ребер: {G.number_of_edges()}")
```

**Возвращает:**
- `G`: NetworkX граф
- `img_contours_mono`: Монохромное изображение контуров

### get_edge_type(node1, node2, cnts, nodes_metadata, wc_eps=30, border_pixel=0)

Определение типа ребра между узлами.

```python
from combra import graph

edge_type = graph.get_edge_type(
    node1, node2,
    cnts,
    nodes_metadata,
    wc_eps=30
)
# Возвращает: 0 (Co), 1 (WC-Co), 2 (WC), 3 (WC-WC)
```

### get_edge_type_labeled(node1, node2, nodes_metadata, line_eps=10)

Определение типа ребра для размеченных данных.

```python
from combra import graph

edge_type = graph.get_edge_type_labeled(
    node1, node2,
    nodes_metadata,
    line_eps=10
)
```

### get_energies(g, cnts, nodes_metadata, entry_nodes, exit_nodes, ...)

Вычисление энергий путей в графе.

```python
from combra import graph

energies = graph.get_energies(
    g,
    cnts,
    nodes_metadata,
    entry_nodes,
    exit_nodes,
    workers=23
)
```

### find_shortest_energy_paths(g, energies_paths, ...)

Поиск кратчайших путей по энергиям.

```python
from combra import graph

paths = graph.find_shortest_energy_paths(
    g,
    energies_paths,
    ...
)
```

### graph_plot(g, img_contours, N=50, M=50, name='graph.jpg', border=30, save=False)

Визуализация графа на изображении.

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

### plot_optimized_energies(...)

Визуализация матриц энергий.

```python
from combra import graph

graph.plot_optimized_energies(
    energies,
    ...
)
```

### plot_paths(g, df, img_aligned, border=30)

Визуализация путей на изображении.

```python
from combra import graph

graph.plot_paths(G, paths_df, img_aligned, border=30)
```

### plot_optimized_paths(g, energies_paths, img_contours_o, param_1=10, param_2=10)

Визуализация оптимизированных путей.

```python
from combra import graph

graph.plot_optimized_paths(
    G,
    energies_paths,
    img_contours_o,
    param_1=10,
    param_2=10
)
```

### draw_tree(img, centres=False, leafs=False, nodes=False, bones=False)

Визуализация дерева на изображении.

```python
from combra import graph

graph.draw_tree(
    img,
    centres=True,
    leafs=True,
    nodes=True
)
```

---

## mvee

Модуль для работы с минимальным объемлющим эллипсоидом (MVEE).

### get_mvee_params(image, tol=0.2)

Вычисление параметров MVEE для изображения.

```python
from combra import mvee

params = mvee.get_mvee_params(image, tol=0.2)
```

### diametr_approx_save(images_path, save_path, types_dict, step, pixel, start=2, end=-3, max_images_num_per_class=None)

Вычисление и сохранение диаметров (MVEE) для датасета.

```python
from combra import mvee, data

images_path = data.example_class_path()
types_dict = {
    'Ultra_Co11': 'средние зерна',
    'Ultra_Co25': 'мелкие зерна'
}

mvee.diametr_approx_save(
    images_path=images_path,
    save_path='mvee_results',
    types_dict=types_dict,
    step=4,
    pixel=50/1000,  # Размер пикселя в мм
    max_images_num_per_class=None
)
```

### plot_beam_base(data, save_name, step, N, M, indices=None, save=False, scatter_size=60, font_size=20)

Визуализация базовых данных MVEE.

```python
from combra import mvee
import json

with open('mvee_results_step_4_beams.json', 'r') as f:
    data = json.load(f)

mvee.plot_beam_base(
    data,
    save_name='mvee_plot',
    step=4,
    N=7,
    M=7,
    save=True
)
```

### plot_angles(data, saved_image_name, step, N, M, indices=None, save=False)

Визуализация углов MVEE.

```python
from combra import mvee

mvee.plot_angles(data, 'angles_plot', step=4, N=7, M=7, save=True)
```

### beams_heatmap(data, step, saved_names, indices=None, bin_max=30, N=7, M=7, font_size=20, save=False, scatter_size=60)

Визуализация тепловой карты MVEE.

```python
from combra import mvee

mvee.beams_heatmap(
    data,
    step=4,
    saved_names=['heatmap'],
    N=7,
    M=7,
    save=True
)
```

### enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)

Визуализация объемлющего эллипса.

```python
from combra import mvee

mvee.enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)
```

---

## areas

Модуль для анализа площадей полигонов.

### plot_polygons_area(data, saved_image_name, step, N, M, indices=None, save=False, start=1, end=None, pixel=50/1000, font_size=20, s=60, log_min_val=-8, min_area_num=10)

Визуализация распределения площадей полигонов.

```python
from combra import areas
import json

with open('areas_data.json', 'r') as f:
    data = json.load(f)

areas.plot_polygons_area(
    data,
    saved_image_name='areas_plot',
    step=5,
    N=10,
    M=10,
    pixel=50/1000,  # Размер пикселя в мм
    save=True
)
```

### plot_polygons_effective_radius(data, saved_image_name, step, N, M, indices=None, save=False, start=1, end=None, ...)

Визуализация эффективных радиусов полигонов.

```python
from combra import areas

areas.plot_polygons_effective_radius(
    data,
    saved_image_name='radius_plot',
    step=5,
    N=10,
    M=10,
    save=True
)
```

---

## approx

Модуль для аппроксимации данных.

### gaussian_fit(x, y, mu=1, sigma=1, amp=1)

Аппроксимация данных одной гауссовой функцией.

```python
from combra import approx
import numpy as np

x = np.linspace(0, 360, 100)
y = some_data  # Ваши данные

mu, sigma, amp = approx.gaussian_fit(x, y, mu=100, sigma=30, amp=1)
print(f"μ={mu:.2f}, σ={sigma:.2f}, A={amp:.2f}")
```

**Возвращает:**
- `mu`: Среднее значение
- `sigma`: Стандартное отклонение
- `amp`: Амплитуда

### gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)

Аппроксимация данных бимодальной гауссовой функцией.

```python
from combra import approx

mus, sigmas, amps = approx.gaussian_fit_bimodal(
    x, y,
    mu1=90, mu2=270,
    sigma1=30, sigma2=30,
    amp1=1, amp2=1
)
```

**Возвращает:**
- `mus`: [mu1, mu2]
- `sigmas`: [sigma1, sigma2]
- `amps`: [amp1, amp2]

### gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1, amp3=1)

Аппроксимация данных термодальной гауссовой функцией.

```python
from combra import approx

mus, sigmas, amps = approx.gaussian_fit_termodal(
    x, y,
    mu1=10, mu2=100, mu3=240,
    sigma1=10, sigma2=30, sigma3=30,
    amp1=1, amp2=1, amp3=1
)
```

### bimodal_gauss_approx(x, y)

Полная бимодальная аппроксимация с возвратом кривой.

```python
from combra import approx

(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# Визуализация
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Данные')
plt.plot(x_gauss, y_gauss, '-', label='Аппроксимация')
plt.legend()
plt.show()
```

**Возвращает:**
- `(x_gauss, y_gauss)`: Координаты аппроксимирующей кривой
- `mus`: [mu1, mu2]
- `sigmas`: [sigma1, sigma2]
- `amps`: [amp1, amp2]

### lin_regr_approx(x, y)

Линейная регрессия.

```python
from combra import approx

slope, intercept = approx.lin_regr_approx(x, y)
```

---

## stats

Модуль для статистической обработки.

### stats_preprocess(array, step)

Предобработка массива для статистического анализа.

```python
from combra import stats

angles = np.array([...])  # Массив углов
x, y = stats.stats_preprocess(angles, step=5)
# x - центры бинов, y - частоты
```

**Параметры:**
- `array` (ndarray): Входной массив данных
- `step` (float): Шаг для гистограммы

**Возвращает:**
- `x`: Центры бинов
- `y`: Частоты

### gaussian(x, mu, sigma, amp=1)

Гауссова функция.

```python
from combra import stats
import numpy as np

x = np.linspace(0, 360, 100)
y = stats.gaussian(x, mu=100, sigma=30, amp=1)
```

### gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1)

Бимодальная гауссова функция.

```python
from combra import stats

y = stats.gaussian_bimodal(
    x,
    mu1=90, mu2=270,
    sigma1=30, sigma2=30,
    amp1=1, amp2=1
)
```

### gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1)

Термодальная гауссова функция.

```python
from combra import stats

y = stats.gaussian_termodal(
    x,
    mu1=10, mu2=100, mu3=240,
    sigma1=10, sigma2=30, sigma3=30,
    amp1=1, amp2=1, amp3=1
)
```

### ellipse(a, b, angle, xc=0, yc=0, num=50)

Генерация координат эллипса.

```python
from combra import stats

coords = stats.ellipse(a=10, b=5, angle=45, xc=0, yc=0, num=50)
```

---

## Полный пример использования

```python
from combra import data, image, angles, approx, stats
from skimage import io
import json

# 1. Загрузка примера данных
images_path = data.example_class_path()

# 2. Обработка одного изображения
img = io.imread('path/to/image.png')
processed = image.image_preprocess(img)

# 3. Вычисление углов
angles_array, contours = angles.get_angles(processed, border_eps=5, tol=3)

# 4. Статистическая обработка
x, y = stats.stats_preprocess(angles_array, step=5)

# 5. Аппроксимация
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# 6. Визуализация
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Данные')
plt.plot(x_gauss, y_gauss, '-', label='Аппроксимация')
plt.xlabel('Угол (градусы)')
plt.ylabel('Частота')
plt.legend()
plt.show()

# 7. Обработка всего датасета
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
```
