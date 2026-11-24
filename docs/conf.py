"""Sphinx configuration for xmm_tools documentation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "xmm_tools"
author = "Zhipeng"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = ["sherpa", "imgui_bundle"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "xmm_tools"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
