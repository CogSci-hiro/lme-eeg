from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "LmeEEG"
author = "Hiro YAMASAKI"
copyright = "2026, Hiro YAMASAKI"
release = importlib.metadata.version("lmeeeg")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["mne"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo-128.png"
html_show_sourcelink = False

html_theme_options = {
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher"],
    "header_links_before_dropdown": 6,
    "navigation_with_keys": True,
    "show_nav_level": 1,
    "secondary_sidebar_items": ["page-toc"],
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
        "text": "LmeEEG",
        "alt_text": "LmeEEG logo",
    },
}

html_context = {
    "default_mode": "light",
}

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
