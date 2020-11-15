import nengo
import numpy as np
import pytest
from nengo.exceptions import ValidationError
from numpy.testing import assert_almost_equal

import nengo_spa as spa
from nengo_spa.modules.transcode import Transcode
from nengo_spa.testing import assert_sp_close


def test_fixed(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer1 = spa.State(vocab=16)
        model.buffer2 = spa.State(vocab=8, subdimensions=8)
        model.input1 = spa.Transcode("A", output_vocab=16)
        model.input2 = spa.Transcode("B", output_vocab=8)
        model.input1 >> model.buffer1
        model.input2 >> model.buffer2
        p1 = nengo.Probe(model.buffer1.output, synapse=0.03)
        p2 = nengo.Probe(model.buffer2.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.1)

    assert_sp_close(
        sim.trange(), sim.data[p1], model.buffer1.vocab.parse("A"), skip=0.08
    )
    assert_sp_close(
        sim.trange(), sim.data[p2], model.buffer2.vocab.parse("B"), skip=0.08
    )


def test_time_varying_encode(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer = spa.State(vocab=16)

        def stimulus(t):
            if t < 0.1:
                return "A"
            elif t < 0.2:
                return model.buffer.vocab.parse("B")
            elif t < 0.3:
                return model.buffer.vocab.parse("C").v
            else:
                return "0"

        model.encode = spa.Transcode(stimulus, output_vocab=16)
        model.encode >> model.buffer

        p = nengo.Probe(model.buffer.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    vocab = model.buffer.vocab

    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("A"), skip=0.08, duration=0.02
    )
    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("B"), skip=0.18, duration=0.02
    )
    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("C"), skip=0.28, duration=0.02
    )
    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("0"), skip=0.38, duration=0.02
    )


def test_encode_with_input(Simulator, seed):
    with spa.Network(seed=seed) as model:
        buffer = spa.State(vocab=16)

        def stimulus(t, x):
            return x[0] * buffer.vocab.parse("A")

        ctrl = nengo.Node(lambda t: t > 0.2)

        encode = spa.Transcode(stimulus, output_vocab=16, size_in=1)
        nengo.Connection(ctrl, encode.input)
        encode >> buffer

        p = nengo.Probe(buffer.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.4)

    vocab = buffer.vocab
    assert_sp_close(sim.trange(), sim.data[p], vocab.parse("0"), duration=0.2)
    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("A"), skip=0.38, duration=0.02
    )


def test_transcode(Simulator, seed):
    def transcode_fn(t, sp):
        assert t < 0.15 or sp.vocab.parse("A").dot(sp) > 0.8
        return "B"

    with spa.Network(seed=seed) as model:
        transcode = Transcode(transcode_fn, input_vocab=16, output_vocab=16)
        spa.sym.A >> transcode
        p = nengo.Probe(transcode.output, synapse=None)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert_sp_close(sim.trange(), sim.data[p], transcode.output_vocab.parse("B"))


def test_passthrough(Simulator, seed):
    with spa.Network(seed=seed) as model:
        passthrough = Transcode(input_vocab=16, output_vocab=16)
        spa.sym.A >> passthrough
        p = nengo.Probe(passthrough.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert_sp_close(
        sim.trange(), sim.data[p], passthrough.output_vocab.parse("A"), skip=0.18
    )


def test_decode(Simulator, seed):
    class OutputFn:
        def __init__(self):
            self.called = False

        def __call__(self, t, v):
            if t > 0.001:
                self.called = True
                assert_almost_equal(v.vocab.parse("A").v, v.v)

    output_fn = OutputFn()

    with spa.Network(seed=seed) as model:
        model.config[nengo.Connection].synapse = nengo.Lowpass(0.0)
        model.output = Transcode(output_fn, input_vocab=16)
        spa.sym.A >> model.output

    with Simulator(model) as sim:
        sim.run(0.01)

    assert output_fn.called


def test_decode_with_output(Simulator, seed):
    def decode_fn(t, v):
        return [t]

    with spa.Network(seed=seed) as model:
        model.decode = Transcode(decode_fn, input_vocab=16)
        p = nengo.Probe(model.decode.output)

    with Simulator(model) as sim:
        sim.run(0.01)

    assert_almost_equal(np.squeeze(sim.data[p]), sim.trange())


def test_decode_size_out(Simulator, seed):
    def decode_fn(t, v):
        return [t]

    with spa.Network(seed=seed) as model:
        model.decode = Transcode(decode_fn, input_vocab=16, size_out=1)
        p = nengo.Probe(model.decode.output)

    with Simulator(model) as sim:
        sim.run(0.01)

    assert_almost_equal(np.squeeze(sim.data[p]), sim.trange())


def test_exception_when_no_vocabularies_are_given():
    with spa.Network():
        with pytest.raises(ValidationError):
            Transcode("A")
        with pytest.raises(ValidationError):
            Transcode(lambda t: "A")


@pytest.mark.parametrize(
    "value",
    [
        "String",
        spa.semantic_pointer.Zero(32),
        spa.sym.Symbol,
        lambda t: "String",
        lambda t: spa.semantic_pointer.Zero(32),
        lambda t: spa.sym.Symbol,
    ],
)
def test_output_types(Simulator, value):
    with spa.Network() as model:
        stim = spa.Transcode(value, output_vocab=32)
        state = spa.State(32)
        stim >> state

    with Simulator(model) as sim:
        sim.run(0.01)
