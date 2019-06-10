import inspect
import os.path

from nengo.conftest import (  # pylint: disable=unused-import
    pytest_configure, pytest_runtest_setup, recorder_dirname, RefSimulator,
    Simulator, seed, rng)
from nengo.utils.testing import Plotter
import pytest

from nengo_spa.algebras.hrr_algebra import HrrAlgebra
from nengo_spa.algebras.vtb_algebra import VtbAlgebra


# Copied from
# https://github.com/nengo/nengo/blob/dfeb4335e0f8720e1248466719ca08be9a5badfd/nengo/conftest.py
# which contains fixes for pytest 4 that are not released at the time of
# writing this. Also, relying on importing this function from Nengo would also
# prevent us from testing earlier Nengo releases that do not contain the fixes.
def parametrize_function_name(request, function_name):
    """Creates a unique name for a test function.

    The unique name accounts for values passed through
    ``pytest.mark.parametrize``.

    This function is used when naming plots saved through the ``plt`` fixture.
    """
    suffixes = []
    if 'parametrize' in request.keywords:
        argnames = []
        for marker in request.keywords.node.iter_markers("parametrize"):
            argnames.extend([x.strip() for names in marker.args[::2]
                             for x in names.split(',')])
        for name in argnames:
            value = request.getfixturevalue(name)
            if inspect.isclass(value):
                value = value.__name__
            suffixes.append('{}={}'.format(name, value))
    return '+'.join([function_name] + suffixes)


class TestConfig(object):
    """Parameters affecting all Nengo SPA testes.

    These are essentially global variables used by py.test to modify aspects
    of the Nengo SPA tests. We collect them in this class to provide a mini
    namespace and to avoid using the ``global`` keyword.

    The values below are defaults. The functions in the remainder of this
    module modify these values accordingly.
    """

    algebras = [HrrAlgebra(), VtbAlgebra()]


def pytest_generate_tests(metafunc):
    if 'algebra' in metafunc.funcargnames:
        metafunc.parametrize('algebra', [a for a in TestConfig.algebras])


def recorder_dirname_with_algebra(request, name):
    dirname = recorder_dirname(request, name)
    if 'algebra' in request.funcargnames:
        algebra = request.getfixturevalue('algebra')
        dirname = os.path.join(dirname, algebra.__name__)
    return dirname


@pytest.fixture
def plt(request):
    """A pyplot-compatible plotting interface.

    Please use this if your test creates plots.

    This will keep saved plots organized in a simulator-specific folder,
    with an automatically generated name. savefig() and close() will
    automatically be called when the test function completes.

    If you need to override the default filename, set `plt.saveas` to
    the desired filename.
    """
    dirname = recorder_dirname_with_algebra(request, 'plots')
    plotter = Plotter(
        dirname, request.module.__name__,
        parametrize_function_name(request, request.function.__name__))
    request.addfinalizer(lambda: plotter.__exit__(None, None, None))
    return plotter.__enter__()
