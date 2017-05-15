import nengo

import nengo_spa as spa
from nengo_spa.modules.transcode import Transcode
from nengo_spa.testing import sp_close


def test_transcode(Simulator, seed):
    def transcode_fn(t, sp, vocab):
        assert t < 0.15 or vocab.parse('A').dot(sp) > 0.8
        return 'B'

    with spa.Network(seed=seed) as model:
        model.transcode = Transcode(
            transcode_fn, input_vocab=16, output_vocab=16)
        spa.Actions('transcode = A').build()
        p = nengo.Probe(model.transcode.output, synapse=None)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert sp_close(sim.trange(), sim.data[p],
                    model.transcode.output_vocab.parse('B'))


def test_passthrough(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.passthrough = Transcode(input_vocab=16, output_vocab=16)
        spa.Actions('passthrough = A').build()
        p = nengo.Probe(model.passthrough.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert sp_close(sim.trange(), sim.data[p],
                    model.passthrough.output_vocab.parse('A'), skip=0.18)
