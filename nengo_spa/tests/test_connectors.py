import nengo
import pytest

import nengo_spa as spa
from nengo_spa.connectors import as_ast_node
from nengo_spa.exceptions import SpaTypeError


@pytest.mark.parametrize(
    "construct_obj",
    (
        lambda: nengo.Node([1.0, 2.0]),
        lambda: [None],
    ),
)
def test_as_ast_node_error(construct_obj):
    with spa.Network():
        with pytest.raises(SpaTypeError):
            as_ast_node(construct_obj())
