combra
======

## Computer vision for composite alloys 

Series of abstracs:

1) [Computer analysis of the cemented carbides’ microstructure (2021)](https://lettersonmaterials.com/en/Readers/Article.aspx?aid=41463)
2) [Topology of WC/Co Interfaces in Cemented Carbides (2023)](https://doi.org/10.3390/ma16165560) 
3) [Generation of Artificial Images of Cross Sections of WC/Co Composite Alloys Using Diffusion Networks (2025)](https://doi.org/10.1134/S1995080225605569) 


Combra [docs](https://dkagramanyan.github.io/wc_cv/)

Installation
------------

- From source (recommended for development):

```bash
pip install -e .
```

Build docs
----------

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) and the
[PyTorch Sphinx theme](https://github.com/pytorch/pytorch_sphinx_theme2). Pages are
written in Markdown (MyST).

Install the docs toolchain:

```bash
pip install -r docs/requirements.txt
```

Build the static site into `public/`:

```bash
python -m sphinx -b html docs public
```

Live-reloading local preview (optional, needs `sphinx-autobuild`):

```bash
pip install sphinx-autobuild
sphinx-autobuild docs public
# open http://127.0.0.1:8000/
```

The site is published to GitHub Pages automatically on every push to `main`
(`.github/workflows/pages.yaml`).

