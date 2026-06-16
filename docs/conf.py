# Configuration file for the Sphinx documentation builder.
#
# Combra documentation — built with Sphinx and the PyTorch Sphinx theme.
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

import pytorch_sphinx_theme2

_THEME_DIR = pytorch_sphinx_theme2.get_html_theme_path()

# -- Project information -----------------------------------------------------

project = "combra"
copyright = "2026, D.G.Kagramanyan"
author = "D.G.Kagramanyan"
release = "0.3.0"
version = "0.3"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "pytorch_sphinx_theme2",
]

# Markdown (MyST) is the source format for every page.
source_suffix = {".md": "markdown"}
master_doc = "index"

# MyST extensions: dollar/AMS math (rendered by MathJax), colon-fence
# admonitions, definition lists, and reST-style field lists inside directives.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "substitution",
]
myst_heading_anchors = 3

templates_path = ["_templates", os.path.join(_THEME_DIR, "templates")]
# Only the Sphinx sources (index, get_started, api/, examples/) are part of the
# build — keep the legacy Hugo tree and scaffolding out.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "QUICKSTART.md",
    "README.md",
    "content/**",
    "layouts/**",
    "archetypes/**",
    "public/**",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "pytorch_sphinx_theme2"
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]
html_title = "combra"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Link the "Edit this page" / "[source]"-style buttons and the GitHub icon
# back to the repository.
html_context = {
    "github_user": "dkagramanyan",
    "github_repo": "wc_cv",
    "github_version": "main",
    "doc_path": "docs",
    "display_github": True,
}

html_theme_options = {
    "logo_text": "combra",
    "collapse_navigation": False,
    "show_prev_next": True,
    "use_edit_page_button": True,
    # Drop the PyTorch/Linux-Foundation chrome — this is the combra project site.
    "show_lf_header": False,
    "show_lf_footer": False,
    "show_pytorch_org_link": False,
    "navigation_with_keys": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/dkagramanyan/combra",
            "icon": "fa-brands fa-github",
        },
    ],
}

# Sphinx domain settings: show typed signatures the way PyTorch docs do.
add_module_names = False
python_use_unqualified_type_names = True
