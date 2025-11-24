---
title: "Датасет: SEMDataset"
weight: 2
---

Класс `SEMDataset` подготавливает кэш предобработанных изображений в `/tmp/<имя_датасета>` и предоставляет удобный доступ к изображениям и их путям.

## Основные возможности

- Автоматическое кэширование предобработанных изображений
- Параллельная обработка с использованием всех доступных CPU ядер
- Проверка целостности кэша и автоматическое обновление при необходимости
- Поддержка структуры папок с классами изображений

## Создание датасета

```python
from combra.data.dataset import SEMDataset

# Создание датасета
# Путь не должен оканчиваться на '/'
dataset = SEMDataset(
    images_folder_path='data/wc_co',
    max_images_num_per_class=100,
    workers=None  # По умолчанию: cpu_count()-1
)
```

**Параметры:**

- `images_folder_path` (str): Корневая папка с подпапками-классами. **Важно:** путь не должен оканчиваться на `/`
- `max_images_num_per_class` (int): Лимит изображений на класс. Если `None`, используются все доступные изображения
- `workers` (int, optional): Количество процессов для параллельной обработки. По умолчанию `cpu_count()-1`

## Структура папок

Датасет ожидает следующую структуру:

```
data/wc_co/
├── Ultra_Co11/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Ultra_Co25/
│   ├── image1.jpg
│   └── ...
└── Ultra_Co8/
    └── ...
```

## Доступ к данным

### Получение изображения

```python
# Получить изображение и путь по индексам класса и изображения
image, path = dataset.__getitem__(class_idx=0, idx=0)

print(f"Путь к изображению: {path}")
print(f"Форма изображения: {image.shape}")
```

### Количество классов

```python
# Количество классов в датасете
num_classes = len(dataset)
print(f"Классов в датасете: {num_classes}")
```

### Итерация по датасету

```python
# Итерация по всем классам и изображениям
for class_idx in range(len(dataset)):
    num_images = dataset.images_paths.shape[1]
    for img_idx in range(num_images):
        image, path = dataset.__getitem__(class_idx, img_idx)
        # Обработка изображения
        print(f"Класс {class_idx}, Изображение {img_idx}: {path}")
```

## Предобработка изображений

Класс автоматически применяет предобработку к каждому изображению:

1. Конвертация в grayscale (если необходимо)
2. Медианная фильтрация
3. Бинаризация методом Otsu
4. Вычисление градиента

### Использование метода предобработки отдельно

```python
from combra.data.dataset import SEMDataset
from skimage import io

# Загрузка изображения
image = io.imread('path/to/image.jpg')

# Предобработка
processed = SEMDataset.preprocess_image(
    image,
    pad=False,      # Добавить границу
    border=30,      # Размер границы в пикселях
    disk=3          # Размер диска для медианного фильтра
)
```

## Кэширование

Датасет автоматически кэширует предобработанные изображения в `/tmp/<имя_датасета>`. 

### Проверка кэша

При создании датасета выполняется проверка:
- Существования кэша
- Соответствия структуры папок
- Количества изображений

Если кэш неполный или устарел, он автоматически пересоздается.

### Очистка кэша

```python
import shutil
import os

cache_dir = '/tmp/wc_co'  # Замените на имя вашего датасета
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Кэш удален")
```

## Примеры использования

### Базовый пример

```python
from combra.data.dataset import SEMDataset

# Создание датасета
dataset = SEMDataset('data/wc_co', max_images_num_per_class=50)

# Получение первого изображения первого класса
image, path = dataset.__getitem__(0, 0)
print(f"Загружено: {path}")
print(f"Размер: {image.shape}")
```

### Обработка всех изображений

```python
from combra.data.dataset import SEMDataset
from combra import angles

dataset = SEMDataset('data/wc_co', max_images_num_per_class=100)

all_angles = []

for class_idx in range(len(dataset)):
    for img_idx in range(dataset.images_paths.shape[1]):
        image, path = dataset.__getitem__(class_idx, img_idx)
        
        # Вычисление углов
        angles_array, _ = angles.get_angles(image, border_eps=5, tol=3)
        all_angles.extend(angles_array)
        
        print(f"Обработано: {path}")

print(f"Всего углов: {len(all_angles)}")
```

### Использование с другими модулями

```python
from combra.data.dataset import SEMDataset
from combra import angles, stats, approx
import matplotlib.pyplot as plt

# Создание датасета
dataset = SEMDataset('data/wc_co', max_images_num_per_class=50)

# Сбор углов из всех изображений
all_angles = []
for class_idx in range(len(dataset)):
    for img_idx in range(dataset.images_paths.shape[1]):
        image, _ = dataset.__getitem__(class_idx, img_idx)
        angles_array, _ = angles.get_angles(image)
        all_angles.extend(angles_array)

# Статистический анализ
x, y = stats.stats_preprocess(all_angles, step=5)

# Аппроксимация
(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)

# Визуализация
plt.plot(x, y, 'o', label='Данные')
plt.plot(x_gauss, y_gauss, '-', label='Аппроксимация')
plt.legend()
plt.show()
```

## Примечания

- Путь к датасету **не должен** оканчиваться на `/`
- Кэш создается однократно и переиспользуется при последующих запусках
- Предобработанные изображения сохраняются в формате PNG
- Все изображения должны иметь одинаковый размер (проверка не выполняется автоматически)

## Методы класса

### `__init__(images_folder_path, max_images_num_per_class=100, workers=None)`

Инициализация датасета.

### `__getitem__(class_idx, idx)`

Получение изображения и пути.

**Параметры:**
- `class_idx` (int): Индекс класса
- `idx` (int): Индекс изображения в классе

**Возвращает:**
- `tuple`: `(image, path)` где `image` - numpy array, `path` - строка с путем

### `__len__()`

Возвращает количество классов в датасете.

### `preprocess_image(image, pad=False, border=30, disk=3)` (classmethod)

Статический метод для предобработки одного изображения.

**Параметры:**
- `image` (ndarray): Входное изображение
- `pad` (bool): Добавить границу
- `border` (int): Размер границы
- `disk` (int): Размер диска для медианного фильтра

**Возвращает:**
- `ndarray`: Предобработанное изображение
