combra
======

## Computer vision for composite alloys 

Series of abstracs:

1) [Computer analysis of the cemented carbides’ microstructure (2021)](https://lettersonmaterials.com/en/Readers/Article.aspx?aid=41463)
2) [Topology of WC/Co Interfaces in Cemented Carbides (2023)](https://doi.org/10.3390/ma16165560) 
3) [Generation of Artificial Images of Cross Sections of WC/Co Composite Alloys Using Diffusion Networks (2025)](https://doi.org/10.1134/S1995080225605569) 


Installation
------------

- From source (recommended for development):

```bash
pip install -e .
```

Build docs
----------

# Быстрый старт - Запуск документации

## Шаг 1: Установите Hugo

### macOS (через Homebrew)
```bash
brew install hugo
```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install hugo

# или скачайте бинарник с https://github.com/gohugoio/hugo/releases
```

### Windows
Скачайте установщик с [официального сайта Hugo](https://gohugo.io/installation/)

## Шаг 2: Установите Go (если еще не установлен)

Go необходим для работы Hugo Modules.

### macOS
```bash
brew install go
```

### Linux/Windows
Скачайте с [официального сайта Go](https://golang.org/dl/)

## Шаг 3: Перейдите в директорию документации

```bash
cd docs
```

## Шаг 4: Установите тему Hextra

```bash
hugo mod get github.com/imfing/hextra
hugo mod tidy
```

## Шаг 5: Запустите локальный сервер

```bash
hugo server
hugo server --ignoreCache
```

## Шаг 6: Откройте браузер

Перейдите по адресу: **http://localhost:1313/**

---

## Альтернативные команды


### Сборка статического сайта
```bash
hugo
```
Результат будет в папке `public/`

### Сборка с минификацией (для продакшена)
```bash
hugo --minify
```

---


