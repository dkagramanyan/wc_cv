---
title: "Граф трещин"
weight: 3
---

Модуль `graph` предоставляет функциональность для построения и анализа графов трещин на изображениях композитных сплавов.

## Обзор

Граф трещин представляет собой ориентированный граф, где:
- **Узлы** соответствуют точкам пересечения или разветвления трещин
- **Ребра** соответствуют сегментам трещин между узлами
- **Типы ребер** классифицируются по материалу: Co, WC-Co, WC, WC-WC

## Пайплайн работы

1. **Предобработка изображения** (`image.image_preprocess`)
2. **Предобработка для графа** (`graph.preprocess_graph_image`) - выделение контуров, подготовка метаданных узлов
3. **Построение графа** (`graph.create_crack_graph`) - создание ориентированного графа
4. **Вычисление энергий** (опционально) (`graph.get_energies`) - расчет энергий путей
5. **Визуализация** (`graph.graph_plot`) - отображение графа на изображении

## Минимальный пример

```python
from combra import graph, image
from skimage import io

# 1. Загрузка и предобработка изображения
img = io.imread('path/to/crack_image.png')
processed = image.image_preprocess(img)

# 2. Предобработка для графа
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        processed,
        r=2,
        border=30,
        border_node_eps=10,
        tol=5
    )
)

# 3. Создание графа
G, img_contours_mono = graph.create_crack_graph(
    img_contours.shape,
    cnts,
    nodes_meta
)

print(f"Узлов в графе: {G.number_of_nodes()}")
print(f"Ребер в графе: {G.number_of_edges()}")

# 4. Визуализация
graph.graph_plot(
    G,
    img_contours_mono,
    N=10,
    M=10,
    name='crack_graph.png',
    save=True
)
```

## Предобработка для графа

### `preprocess_graph_image()`

Подготавливает изображение для построения графа: выделяет контуры, находит узлы, определяет входные и выходные узлы.

```python
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        image,
        r=2,                    # Радиус для обработки
        border=30,              # Размер границы
        border_node_eps=10,     # Эпсилон для узлов на границе
        tol=5,                  # Точность упрощения контуров
        disk=5,                 # Размер диска для морфологии
        labeled_cnts=False,     # Использовать размеченные контуры
        labels=False            # Использовать метки
    )
)
```

**Параметры:**
- `image` (ndarray): Предобработанное изображение
- `r` (int): Радиус для обработки узлов
- `border` (int): Размер границы в пикселях
- `border_node_eps` (int): Порог для определения узлов на границе
- `tol` (int): Точность упрощения контуров (алгоритм Дугласа-Пекера)
- `disk` (int): Размер диска для морфологических операций
- `labeled_cnts` (bool): Использовать размеченные контуры
- `labels` (bool): Использовать метки классов

**Возвращает:**
- `entry_nodes`: Список индексов входных узлов
- `exit_nodes`: Список индексов выходных узлов
- `img_contours`: Изображение с контурами
- `img_marked`: Размеченное изображение
- `cnts`: Список контуров
- `nodes_meta`: Словарь с метаданными узлов

## Построение графа

### `create_crack_graph()`

Создает ориентированный граф трещин на основе контуров и метаданных узлов.

```python
G, img_contours_mono = graph.create_crack_graph(
    img_shape,
    cnts,
    nodes_metadata,
    # Дополнительные параметры...
)
```

**Параметры:**
- `img_shape`: Форма изображения (tuple)
- `cnts`: Список контуров
- `nodes_metadata`: Словарь с метаданными узлов

**Возвращает:**
- `G`: NetworkX ориентированный граф
- `img_contours_mono`: Монохромное изображение контуров

**Структура графа:**
- Узлы содержат атрибуты: координаты, тип узла
- Ребра содержат атрибуты: тип ребра, длина, энергия (если вычислена)

## Типы ребер

### `get_edge_type()`

Определяет тип ребра между двумя узлами на основе анализа пикселей вдоль линии.

```python
edge_type = graph.get_edge_type(
    node1,
    node2,
    cnts,
    nodes_metadata,
    wc_eps=30,
    border_pixel=0
)
```

**Типы ребер:**
- `0`: Co (кобальт)
- `1`: WC-Co (граница вольфрама-кобальта)
- `2`: WC (вольфрам-карбид)
- `3`: WC-WC (граница вольфрама-вольфрама)

### `get_edge_type_labeled()`

Определение типа ребра для размеченных данных.

```python
edge_type = graph.get_edge_type_labeled(
    node1,
    node2,
    nodes_metadata,
    line_eps=10
)
```

## Вычисление энергий

### `get_energies()`

Вычисляет энергии путей в графе для различных конфигураций.

```python
energies = graph.get_energies(
    g,
    cnts,
    nodes_metadata,
    entry_nodes,
    exit_nodes,
    workers=23
)
```

**Параметры:**
- `g`: NetworkX граф
- `cnts`: Список контуров
- `nodes_metadata`: Метаданные узлов
- `entry_nodes`: Входные узлы
- `exit_nodes`: Выходные узлы
- `workers`: Количество процессов

**Возвращает:**
- Словарь с энергиями для различных конфигураций

### `find_shortest_energy_paths()`

Находит кратчайшие пути в графе на основе вычисленных энергий.

```python
paths = graph.find_shortest_energy_paths(
    g,
    energies_paths,
    ...
)
```

## Визуализация

### `graph_plot()`

Визуализирует граф на изображении контуров.

```python
graph.graph_plot(
    g,
    img_contours,
    N=50,              # Количество строк в сетке
    M=50,              # Количество столбцов в сетке
    name='graph.jpg',  # Имя файла для сохранения
    border=30,         # Размер границы
    save=False         # Сохранить изображение
)
```

**Пример:**
```python
graph.graph_plot(
    G,
    img_contours_mono,
    N=10,
    M=10,
    name='crack_graph.png',
    border=30,
    save=True
)
```

### `plot_optimized_energies()`

Визуализирует матрицы энергий для различных конфигураций.

```python
graph.plot_optimized_energies(
    energies,
    ...
)
```

### `plot_paths()`

Визуализирует пути на выровненном изображении.

```python
graph.plot_paths(
    g,
    df,           # DataFrame с путями
    img_aligned,  # Выровненное изображение
    border=30
)
```

### `plot_optimized_paths()`

Визуализирует оптимизированные пути.

```python
graph.plot_optimized_paths(
    g,
    energies_paths,
    img_contours_o,
    param_1=10,
    param_2=10
)
```

### `draw_tree()`

Визуализирует дерево на изображении.

```python
graph.draw_tree(
    img,
    centres=False,  # Показывать центры
    leafs=False,    # Показывать листья
    nodes=False,    # Показывать узлы
    bones=False     # Показывать кости
)
```

## Полный пример

```python
from combra import graph, image, contours
from skimage import io
import networkx as nx

# 1. Загрузка и предобработка
img = io.imread('crack_image.png')
processed = image.image_preprocess(img)

# 2. Предобработка для графа
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        processed,
        r=2,
        border=30,
        border_node_eps=10,
        tol=5
    )
)

# 3. Создание графа
G, img_contours_mono = graph.create_crack_graph(
    img_contours.shape,
    cnts,
    nodes_meta
)

# 4. Анализ графа
print(f"Узлов: {G.number_of_nodes()}")
print(f"Ребер: {G.number_of_edges()}")
print(f"Входных узлов: {len(entry_nodes)}")
print(f"Выходных узлов: {len(exit_nodes)}")

# 5. Вычисление энергий (опционально)
energies = graph.get_energies(
    G,
    cnts,
    nodes_meta,
    entry_nodes,
    exit_nodes,
    workers=4
)

# 6. Визуализация
graph.graph_plot(
    G,
    img_contours_mono,
    N=10,
    M=10,
    name='crack_graph.png',
    save=True
)

# 7. Анализ типов ребер
edge_types = {}
for u, v, data in G.edges(data=True):
    edge_type = data.get('type', 'unknown')
    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

print("Распределение типов ребер:")
for edge_type, count in edge_types.items():
    type_names = {0: 'Co', 1: 'WC-Co', 2: 'WC', 3: 'WC-WC'}
    print(f"  {type_names.get(edge_type, 'Unknown')}: {count}")
```

## Работа с NetworkX

Поскольку граф является объектом NetworkX, можно использовать все стандартные функции NetworkX:

```python
import networkx as nx

# Поиск кратчайших путей
if nx.has_path(G, source=entry_nodes[0], target=exit_nodes[0]):
    path = nx.shortest_path(G, source=entry_nodes[0], target=exit_nodes[0])
    print(f"Кратчайший путь: {path}")

# Вычисление центральности
centrality = nx.degree_centrality(G)
print(f"Наиболее центральный узел: {max(centrality, key=centrality.get)}")

# Поиск компонент связности
components = list(nx.weakly_connected_components(G))
print(f"Компонент связности: {len(components)}")
```

## Параметры и настройки

### Рекомендуемые параметры

Для большинства случаев подходят следующие параметры:

```python
# Предобработка
processed = image.image_preprocess(img)

# Предобработка для графа
entry_nodes, exit_nodes, img_contours, img_marked, cnts, nodes_meta = (
    graph.preprocess_graph_image(
        processed,
        r=2,                # Стандартное значение
        border=30,          # Зависит от размера изображения
        border_node_eps=10, # Стандартное значение
        tol=5,              # Баланс между точностью и производительностью
        disk=5              # Стандартное значение
    )
)
```

### Настройка для разных типов изображений

- **Высокое разрешение**: увеличьте `border` и `border_node_eps`
- **Много шума**: увеличьте `disk` для морфологии
- **Сложные контуры**: уменьшите `tol` для большей точности

## Примечания

- Граф является ориентированным (directed)
- Узлы на границе изображения автоматически определяются как входные/выходные
- Типы ребер определяются на основе анализа пикселей вдоль линии между узлами
- Для больших изображений рекомендуется использовать параллельную обработку (`workers` параметр)
