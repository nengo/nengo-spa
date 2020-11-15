import os

import _pytest.capture
import pytest
from nbformat import read as read_nb
from nengo.utils.stdlib import execfile

# Monkeypatch _pytest.capture.DontReadFromInput
#  If we don't do this, importing IPython will choke as it reads the current
#  sys.stdin to figure out the encoding it will use; pytest installs
#  DontReadFromInput as sys.stdin to capture output.
#  Running with -s option doesn't have this issue, but this monkeypatch
#  doesn't have any side effects, so it's fine.
_pytest.capture.DontReadFromInput.encoding = "utf-8"
_pytest.capture.DontReadFromInput.write = lambda: None
_pytest.capture.DontReadFromInput.flush = lambda: None

example_dir = "docs/examples"

too_slow = [
    "intro",
    "intro-coming-from-legacy-spa",
    "question",
    "question-control",
    "question-memory",
    "spa-parser",
    "spa-sequence",
    "spa-sequence-routed",
]

all_examples, slow_examples, fast_examples = [], [], []


def load_example(example):
    with open(example + ".ipynb", "r") as f:
        nb = read_nb(f, 4)
    return nb


for subdir, _, files in os.walk(example_dir):
    if (os.path.sep + ".") in subdir:
        continue
    files = [f for f in files if f.endswith(".ipynb")]
    examples = [os.path.join(subdir, os.path.splitext(f)[0]) for f in files]
    all_examples.extend(examples)
    slow_examples.extend(
        [e for e, f in zip(examples, files) if os.path.splitext(f)[0] in too_slow]
    )
    fast_examples.extend(
        [e for e, f in zip(examples, files) if os.path.splitext(f)[0] not in too_slow]
    )

# os.walk goes in arbitrary order, so sort after the fact to keep pytest happy
all_examples.sort()
slow_examples.sort()
fast_examples.sort()


def assert_noexceptions(nb_file, tmpdir):
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("IPython", minversion="1.0")
    pytest.importorskip("jinja2")
    from nengo.utils.ipython import export_py

    nb = load_example(nb_file)
    pyfile = "%s.py" % tmpdir.join(os.path.basename(nb_file))
    export_py(nb, pyfile)
    execfile(pyfile, {})
    plt.close("all")


@pytest.mark.example
@pytest.mark.parametrize("nb_file", fast_examples)
def test_fast_noexceptions(nb_file, tmpdir):
    """Ensure that no cells raise an exception."""
    assert_noexceptions(nb_file, tmpdir)


@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize("nb_file", slow_examples)
def test_slow_noexceptions(nb_file, tmpdir):
    """Ensure that no cells raise an exception."""
    assert_noexceptions(nb_file, tmpdir)


def iter_cells(nb_file, cell_type="code"):
    nb = load_example(nb_file)

    if nb.nbformat <= 3:
        cells = []
        for ws in nb.worksheets:
            cells.extend(ws.cells)
    else:
        cells = nb.cells

    for cell in cells:
        if cell.cell_type == cell_type:
            yield cell


@pytest.mark.example
@pytest.mark.parametrize("nb_file", all_examples)
def test_no_signature(nb_file):
    nb = load_example(nb_file)
    assert "signature" not in nb.metadata, "Notebook has signature"


@pytest.mark.example
@pytest.mark.parametrize("nb_file", all_examples)
def test_no_outputs(nb_file):
    """Ensure that no cells have output."""
    pytest.importorskip("IPython", minversion="1.0")

    for cell in iter_cells(nb_file):
        assert cell.outputs == [], "Cell outputs not cleared"
