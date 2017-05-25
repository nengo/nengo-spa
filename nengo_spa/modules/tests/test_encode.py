import nengo

import nengo_spa as spa
from nengo_spa.testing import sp_close


def test_fixed(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer1 = spa.State(vocab=16)
        model.buffer2 = spa.State(vocab=8, subdimensions=8)
        model.input1 = spa.Encode('A', vocab=16)
        model.input2 = spa.Encode('B', vocab=8)
        spa.Actions('buffer1 = input1', 'buffer2 = input2').build()
        p1 = nengo.Probe(model.buffer1.output, synapse=0.03)
        p2 = nengo.Probe(model.buffer2.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.1)

    input1, vocab1 = model.get_network_input('buffer1')
    input2, vocab2 = model.get_network_input('buffer2')

    assert sp_close(sim.trange(), sim.data[p1], vocab1.parse('A'), skip=0.08)
    assert sp_close(sim.trange(), sim.data[p2], vocab2.parse('B'), skip=0.08)


def test_time_varying(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer = spa.State(vocab=16)

        def stimulus(t):
            if t < 0.1:
                return 'A'
            elif t < 0.2:
                return model.buffer.vocab.parse('B')
            elif t < 0.3:
                return model.buffer.vocab.parse('C').v
            else:
                return '0'

        model.encode = spa.Encode(stimulus, vocab=16)
        spa.Actions('buffer = encode').build()

        p = nengo.Probe(model.buffer.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    vocab = model.buffer.vocab

    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('A'), skip=0.08, duration=0.02)
    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('B'), skip=0.18, duration=0.02)
    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('C'), skip=0.28, duration=0.02)
    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('0'), skip=0.38, duration=0.02)


def test_with_input(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.buffer = spa.State(vocab=16)

        def stimulus(t, x):
            return x[0] * model.buffer.vocab.parse('A')

        ctrl = nengo.Node(lambda t: t > 0.2)

        model.encode = spa.Encode(stimulus, vocab=16, size_in=1)
        nengo.Connection(ctrl, model.encode.input)
        spa.Actions('buffer = encode').build()

        p = nengo.Probe(model.buffer.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.4)

    vocab = model.buffer.vocab
    assert sp_close(sim.trange(), sim.data[p], vocab.parse('0'), duration=0.2)
    assert sp_close(
        sim.trange(), sim.data[p], vocab.parse('A'), skip=.38, duration=0.02)
