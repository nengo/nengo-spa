import nengo
import numpy as np
from numpy.testing import assert_almost_equal

import nengo_spa as spa
from nengo_spa.modules.decode import Decode


def test_decode(Simulator, seed):
    class OutputFn(object):
        def __init__(self):
            self.called = False

        def __call__(self, t, v, vocab):
            if t > 0.001:
                self.called = True
                assert_almost_equal(vocab.parse('A').v, v.v)

    output_fn = OutputFn()

    with spa.Network(seed=seed) as model:
        model.config[nengo.Connection].synapse = nengo.Lowpass(0.)
        model.output = Decode(output_fn, vocab=16)
        spa.Actions('output = A').build()

    with Simulator(model) as sim:
        sim.run(0.01)

    assert output_fn.called


def test_with_output(Simulator, seed):
    def decode_fn(t, v, vocab):
        return [t]

    with spa.Network(seed=seed) as model:
        model.decode = Decode(decode_fn, vocab=16)
        p = nengo.Probe(model.decode.output)

    with Simulator(model) as sim:
        sim.run(0.01)

    assert_almost_equal(np.squeeze(sim.data[p]), sim.trange())


def test_size_out(Simulator, seed):
    def decode_fn(t, v, vocab):
        return [t]

    with spa.Network(seed=seed) as model:
        model.decode = Decode(decode_fn, vocab=16, size_out=1)
        p = nengo.Probe(model.decode.output)

    with Simulator(model) as sim:
        sim.run(0.01)

    assert_almost_equal(np.squeeze(sim.data[p]), sim.trange())
