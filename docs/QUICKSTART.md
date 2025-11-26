# Quick Start - Running the Documentation

## Step 1: Install Hugo

### macOS (via Homebrew)
```bash
brew install hugo
```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install hugo

# or download binary from https://github.com/gohugoio/hugo/releases
```

### Windows
Download installer from [Hugo official website](https://gohugo.io/installation/)

## Step 2: Install Go (if not already installed)

Go is required for Hugo Modules to work.

### macOS
```bash
brew install go
```

### Linux/Windows
Download from [Go official website](https://golang.org/dl/)

## Step 3: Navigate to the documentation directory

```bash
cd docs
```

## Step 4: Install Hextra theme

```bash
hugo mod get github.com/imfing/hextra
hugo mod tidy
```

## Step 5: Start local server

```bash
hugo server
hugo server --noHTTPCache --ignoreCache --disableFastRender
```

## Step 6: Open browser

Navigate to: **http://localhost:1313/**

---

## Alternative commands


### Build static site
```bash
hugo
```
Result will be in the `public/` folder

### Build with minification (for production)
```bash
hugo --minify
```

---
