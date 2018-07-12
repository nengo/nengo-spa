from nengo.conftest import (  # pylint: disable=unused-import
    pytest_configure, pytest_runtest_setup, RefSimulator, Simulator, plt, seed,
    rng)
import pytest

from nengo_spa.algebras.cconv import CircularConvolutionAlgebra
from nengo_spa.algebras.vtb import VtbAlgebra


class TestConfig(object):
    """Parameters affecting all Nengo SPA testes.

    These are essentially global variables used by py.test to modify aspects
    of the Nengo SPA tests. We collect them in this class to provide a mini
    namespace and to avoid using the ``global`` keyword.

    The values below are defaults. The functions in the remainder of this
    module modify these values accordingly.
    """

    algebras = [CircularConvolutionAlgebra, VtbAlgebra]


def pytest_generate_tests(metafunc):
    marks = [
        getattr(pytest.mark, m.name)(*m.args, **m.kwargs)
        for m in getattr(metafunc.function, 'pytestmark', [])]

    if 'algebra' in metafunc.funcargnames:
        metafunc.parametrize('algebra', [a for a in TestConfig.algebras])
