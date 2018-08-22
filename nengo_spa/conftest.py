import os.path

from nengo.conftest import (  # pylint: disable=unused-import
    parametrize_function_name, pytest_configure, pytest_runtest_setup,
    recorder_dirname, RefSimulator, Simulator, seed, rng)
from nengo.utils.testing import Plotter
import pytest

from nengo_spa.algebras.hrr_algebra import HrrAlgebra
from nengo_spa.algebras.vtb_algebra import VtbAlgebra


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
