---
title: "Установка"
weight: 6
---

## Установка из исходников

Для работы с библиотекой `combra` склонируйте репозиторий и установите пакет в режиме разработки:

```bash
git clone https://github.com/dkagramanyan/wc_cv
cd wc_cv/combra
pip install -e .
```

Требуемая версия Python - 3.8 или выше.

## Зависимости

Основные зависимости устанавливаются автоматически при установке пакета. Основные библиотеки:
- `numpy` - работа с массивами
- `scikit-image` - обработка изображений
- `networkx` - работа с графами
- `lmfit` - аппроксимация функций
- `opencv-python` - компьютерное зрение
- `mpire` - параллельная обработка

## Проверка установки

```python
import combra
print(f"combra version: {combra.__version__}")

# Проверка основных модулей
from combra import data, image, angles, graph
print("Все модули успешно импортированы!")
```

Тестовые изображения находятся в хранилище s3 по адресам

1. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co11.jpg 
2. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co15.jpg
3. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co25.jpg
4. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co6_2.jpg
5. https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co8.jpg

