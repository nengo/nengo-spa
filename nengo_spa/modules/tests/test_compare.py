import nengo

import nengo_spa as spa


def test_basic():
    with spa.Network():
        compare = spa.Compare(vocab=16)

    vocab_a = spa.Network.get_input_vocab(compare.input_a)
    vocab_b = spa.Network.get_input_vocab(compare.input_b)
    # all inputs should share the same vocab
    assert vocab_a is vocab_b
    assert vocab_a.dimensions == 16
    # output should have no vocab
    assert spa.Network.get_output_vocab(compare.output) is None


def test_run(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.compare = spa.Compare(vocab=16)
        model.compare.vocab.populate("A; B")

        def inputA(t):
            if 0 <= t < 0.1:
                return "A"
            else:
                return "B"

        model.input = spa.Transcode(inputA, output_vocab=16)
        model.input >> model.compare.input_a
        spa.sym.A >> model.compare.input_b

        p = nengo.Probe(model.compare.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert sim.data[p][100] > 0.8
    assert sim.data[p][199] < 0.2
