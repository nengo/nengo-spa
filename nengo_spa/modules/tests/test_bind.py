import nengo
import pytest
from nengo.utils.numpy import rms

import nengo_spa as spa
from nengo_spa.testing import assert_sp_close


def test_basic():
    with spa.Network():
        bind = spa.Bind(vocab=16)

    # all inputs and outputs should share the same vocab
    vocab_a = spa.Network.get_input_vocab(bind.input_left)
    vocab_b = spa.Network.get_input_vocab(bind.input_right)
    assert vocab_a is vocab_b
    assert vocab_a.dimensions == 16
    assert vocab_b.dimensions == 16


def test_run(Simulator, algebra, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        model.bind = spa.Bind(vocab)

        def inputA(t):
            if 0 <= t < 0.1:
                return "A"
            else:
                return "B"

        model.input = spa.Transcode(inputA, output_vocab=vocab)
        model.input >> model.bind.input_left
        spa.sym.A >> model.bind.input_right

        p = nengo.Probe(model.bind.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    error = rms(vocab.parse("(B*A).normalized()").v - sim.data[p][-1])
    assert error < 0.15

    error = rms(vocab.parse("(A*A).normalized()").v - sim.data[p][100])
    assert error < 0.15


@pytest.mark.parametrize("side", ("left", "right"))
def test_unbind(Simulator, algebra, side, seed, rng):
    vocab = spa.Vocabulary(64, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        model.bind = spa.Bind(
            vocab, unbind_left=(side == "left"), unbind_right=(side == "right")
        )

        if side == "left":
            spa.sym.B >> model.bind.input_left
            spa.sym.B * spa.sym.A >> model.bind.input_right
        elif side == "right":
            spa.sym.A * spa.sym.B >> model.bind.input_left
            spa.sym.B >> model.bind.input_right
        else:
            raise ValueError("Invalid 'side' value.")

        p = nengo.Probe(model.bind.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("A * B * B.rinv()"), skip=0.15, atol=0.3
    )
