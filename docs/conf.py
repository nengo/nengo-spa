# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set
# to its containing dir.

import os
import sys

try:
    import nengo_spa
    import sphinx_rtd_theme
except ImportError:
    print("To build the documentation, nengo_spa and sphinx_rtd_theme "
          "must be installed in the current environment. Please install these "
          "and their requirements first. A virtualenv is recommended!")
    sys.exit(1)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'nengo.utils.docutils',
    'nbsphinx',
]

default_role = 'py:obj'
numfig = True

# -- sphinx.ext.autodoc
autoclass_content = 'both'  # class and __init__ docstrings are concatenated
autodoc_default_flags = ['members']
autodoc_member_order = 'bysource'  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    'nengo': ('https://www.nengo.ai/nengo/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
}

# -- sphinx.ext.todo
todo_include_todos = True

# -- nbsphinx
nbsphinx_allow_errors = False
nbsphinx_timeout = 300
nbsphinx_execute = 'always'

# -- sphinx
exclude_patterns = ['_build', 'examples/.ipynb_checkpoints']
source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'

# Need to include https Mathjax path for sphinx < v1.3
mathjax_path = ("https://cdn.mathjax.org/mathjax/latest/MathJax.js"
                "?config=TeX-AMS-MML_HTMLorMML")

project = u'Nengo SPA'
authors = u'Applied Brain Research'
copyright = nengo_spa.__copyright__
version = '.'.join(nengo_spa.__version__.split('.')[:2])  # Short X.Y version
release = nengo_spa.__version__  # Full version, with tags
pygments_style = 'default'

# -- Options for HTML output --------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = "Nengo SPA {0} docs".format(release)
html_static_path = ['_static']
html_favicon = os.path.join('_static', 'favicon.ico')
html_logo = os.path.join('_static', 'square-light.svg')
html_context = {
    'css_files': [os.path.join('_static', 'custom.css')],
}
html_use_smartypants = True
htmlhelp_basename = 'Nengodoc'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
html_show_sphinx = False

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
}

latex_documents = [
    # (source start file, target, title, author, documentclass [howto/manual])
    ('index', 'nengo.tex', html_title, authors, 'manual'),
]

# -- Options for manual page output -------------------------------------------

man_pages = [
    # (source start file, name, description, authors, manual section).
    ('index', 'nengo', html_title, [authors], 1)
]

# -- Options for Texinfo output -----------------------------------------------

texinfo_documents = [
    # (source start file, target, title, author, dir menu entry,
    #  description, category)
    ('index', 'nengo', html_title, authors, 'Nengo',
     'Large-scale neural simulation in Python', 'Miscellaneous'),
]
