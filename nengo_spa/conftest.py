import pytest

try:
    from pytest_nengo import (  # pylint: disable=unused-import
        pytest_configure,
        pytest_runtest_setup,
    )
except ImportError:
    import nengo

    @pytest.fixture(scope="session")
    def Simulator(request):
        return nengo.Simulator


from nengo_spa.algebras.hrr_algebra import HrrAlgebra
from nengo_spa.algebras.tvtb_algebra import TvtbAlgebra
from nengo_spa.algebras.vtb_algebra import VtbAlgebra


class TestConfig:
    """Parameters affecting all NengoSPA testes.

    These are essentially global variables used by py.test to modify aspects
    of the NengoSPA tests. We collect them in this class to provide a mini
    namespace and to avoid using the ``global`` keyword.

    The values below are defaults. The functions in the remainder of this
    module modify these values accordingly.
    """

    algebras = [HrrAlgebra(), VtbAlgebra(), TvtbAlgebra()]


def pytest_generate_tests(metafunc):
    if "algebra" in metafunc.fixturenames:
        metafunc.parametrize(
            "algebra",
            [pytest.param(a, id=a.__class__.__name__) for a in TestConfig.algebras],
        )


def check_sidedness(algebra, method_name, sidedness):
    method = getattr(algebra, method_name)
    if (
        hasattr(method, "supported_sidedness")
        and sidedness not in method.supported_sidedness
    ):
        pytest.xfail(
            f"Algebra {algebra.__class__.__name__} does not have a "
            f"{sidedness} {method_name}."
        )
