# -*- coding: utf-8 -*-
#
# Automatically generated by nengo-bones, do not edit this file directly

import pathlib

import nengo_spa

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "nbsphinx",
    "nengo_sphinx_theme",
    "nengo_sphinx_theme.ext.backoff",
    "nengo_sphinx_theme.ext.redirects",
    "nengo_sphinx_theme.ext.sourcelinks",
    "notfound.extension",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

# -- sphinx.ext.autodoc
autoclass_content = "both"  # class and __init__ docstrings are concatenated
autodoc_default_options = {"members": None}
autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.doctest
doctest_global_setup = """
import nengo_spa
"""

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "nengo": ("https://www.nengo.ai/nengo/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
}

# -- sphinx.ext.todo
todo_include_todos = True

# -- nbsphinx
nbsphinx_timeout = -1

# -- notfound.extension
notfound_template = "404.html"
notfound_urls_prefix = "/nengo-spa/"

# -- numpydoc config
numpydoc_show_class_members = False

# -- nengo_sphinx_theme.ext.sourcelinks
sourcelinks_module = "nengo_spa"
sourcelinks_url = "https://github.com/nengo/nengo-spa"

# -- sphinx
nitpicky = True
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
]
linkcheck_timeout = 30
source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
linkcheck_ignore = [r"http://localhost:\d+"]
linkcheck_anchors = True
default_role = "py:obj"
pygments_style = "sphinx"
user_agent = "nengo_spa"

project = "NengoSPA"
authors = "Applied Brain Research"
copyright = "2013-2023 Applied Brain Research"
version = ".".join(nengo_spa.__version__.split(".")[:2])  # Short X.Y version
release = nengo_spa.__version__  # Full version, with tags

# -- HTML output
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "nengo_sphinx_theme"
html_title = f"NengoSPA {release} docs"
htmlhelp_basename = "NengoSPA"
html_last_updated_fmt = ""  # Default output format (suppressed)
html_show_sphinx = False
html_favicon = str(pathlib.Path("_static", "favicon.ico"))
html_theme_options = {
    "nengo_logo": "nengo-spa-full-light.svg",
    "nengo_logo_color": "#d40000",
    "analytics": """
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-GT8XEDLTMJ"></script>
        <script>
         window.dataLayer = window.dataLayer || [];
         function gtag(){dataLayer.push(arguments);}
         gtag('js', new Date());
         gtag('config', 'G-GT8XEDLTMJ');
        </script>
        <!-- End Google tag (gtag.js) -->
        <!-- Matomo -->
        <script>
         var _paq = window._paq = window._paq || [];
         _paq.push(["setDocumentTitle", document.domain + "/" + document.title]);
         _paq.push(["setCookieDomain", "*.appliedbrainresearch.com"]);
         _paq.push(["setDomains", ["*.appliedbrainresearch.com","*.edge.nengo.ai","*.forum.nengo.ai","*.nengo.ai"]]);
         _paq.push(["enableCrossDomainLinking"]);
         _paq.push(["setDoNotTrack", true]);
         _paq.push(['trackPageView']);
         _paq.push(['enableLinkTracking']);
         (function() {
           var u="https://appliedbrainresearch.matomo.cloud/";
           _paq.push(['setTrackerUrl', u+'matomo.php']);
           _paq.push(['setSiteId', '3']);
           var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
           g.async=true; g.src='//cdn.matomo.cloud/appliedbrainresearch.matomo.cloud/matomo.js'; s.parentNode.insertBefore(g,s);
         })();
        </script>
        <!-- End Matomo Code -->
    """,
}
html_redirects = [
    ("dev_syntax.html", "dev-syntax.html"),
    ("examples/associative_memory.html", "examples/associative-memory.html"),
    ("examples/custom_module.html", "examples/custom-module.html"),
    (
        "examples/intro_coming_from_legacy_spa.html",
        "examples/intro-coming-from-legacy-spa.html",
    ),
    ("examples/question_control.html", "examples/question-control.html"),
    ("examples/question_memory.html", "examples/question-memory.html"),
    ("examples/spa_parser.html", "examples/spa-parser.html"),
    ("examples/spa_sequence_routed.html", "examples/spa-sequence-routed.html"),
    ("examples/spa_sequence.html", "examples/spa-sequence.html"),
    ("examples/vocabulary_casting.html", "examples/vocabulary-casting.html"),
    ("getting_started.html", "getting-started.html"),
    ("user_guide.html", "user-guide.html"),
    ("user_guide/algebras.html", "user-guide/algebras.html"),
    ("user_guide/spa_intro.html", "user-guide/spa-intro.htm"),
]
