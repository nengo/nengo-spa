import nengo
import pytest

import nengo_spa as spa
from nengo_spa.algebras.vtb_algebra import VtbAlgebra
from nengo_spa.networks.vtb import VTB
from nengo_spa.testing import assert_sp_close


def test_bind(Simulator, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=VtbAlgebra())
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        vtb = VTB(100, 16)
        nengo.Connection(nengo.Node(vocab["A"].v), vtb.input_left)
        nengo.Connection(nengo.Node(vocab["B"].v), vtb.input_right)
        p = nengo.Probe(vtb.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert_sp_close(sim.trange(), sim.data[p], vocab.parse("A*B"), skip=0.15, atol=0.3)


@pytest.mark.parametrize("side", ("left", "right"))
def test_unbind(Simulator, side, seed, rng):
    vocab = spa.Vocabulary(36, pointer_gen=rng, algebra=VtbAlgebra())
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        vtb = VTB(100, 36, unbind_left=(side == "left"), unbind_right=(side == "right"))

        if side == "left":
            left = nengo.Node(vocab["B"].v)
            right = nengo.Node(vocab.parse("B*A").v)
        elif side == "right":
            left = nengo.Node(vocab.parse("A*B").v)
            right = nengo.Node(vocab["B"].v)
        else:
            raise ValueError("Invalid 'side' value.")

        nengo.Connection(left, vtb.input_left)
        nengo.Connection(right, vtb.input_right)

        p = nengo.Probe(vtb.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("A * B * B.rinv()"), skip=0.15, atol=0.3
    )
