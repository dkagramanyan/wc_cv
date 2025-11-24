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
