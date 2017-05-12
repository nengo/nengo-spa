import nengo
from numpy.testing import assert_almost_equal

import nengo_spa as spa
from nengo_spa.modules.output import Output


def test_output(Simulator, seed):
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
        model.output = Output(output_fn, vocab=16)
        spa.Actions('output = A').build()

    with Simulator(model) as sim:
        sim.run(0.01)

    assert output_fn.called
