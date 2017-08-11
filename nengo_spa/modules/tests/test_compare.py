import nengo
import nengo_spa as spa


def test_basic():
    with spa.Network() as model:
        model.compare = spa.Compare(vocab=16)

    inputA = model.get_network_input('compare.input_a')
    inputB = model.get_network_input('compare.input_b')
    output = model.get_network_output('compare')
    # all nodes should be acquired correctly
    assert inputA[0] is model.compare.input_a
    assert inputB[0] is model.compare.input_b
    assert output[0] is model.compare.output
    # all inputs should share the same vocab
    assert inputA[1] is inputB[1]
    assert inputA[1].dimensions == 16
    # output should have no vocab
    assert output[1] is None


def test_run(Simulator, seed):
    with spa.Network(seed=seed) as model:
        model.compare = spa.Compare(vocab=16)
        model.compare.vocab.populate('A; B')

        def inputA(t):
            if 0 <= t < 0.1:
                return 'A'
            else:
                return 'B'

        model.input = spa.Transcode(inputA, output_vocab=16)
        spa.Actions((
            'compare.input_a = input',
            'compare.input_b = A'
        ))

    compare, vocab = model.get_network_output('compare')

    with model:
        p = nengo.Probe(compare, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert sim.data[p][100] > 0.8
    assert sim.data[p][199] < 0.2
