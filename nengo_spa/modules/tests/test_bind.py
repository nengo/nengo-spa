import numpy as np

import nengo
import nengo_spa as spa
from nengo_spa.vocab import VocabularyMap
from nengo.utils.numpy import rmse


def test_basic():
    with spa.Network():
        bind = spa.Bind(vocab=16)

    # all inputs and outputs should share the same vocab
    vocab_a = spa.Network.get_input_vocab(bind.input_a)
    vocab_b = spa.Network.get_input_vocab(bind.input_b)
    assert vocab_a is vocab_b
    assert vocab_a.dimensions == 16
    assert vocab_b.dimensions == 16


def test_run(Simulator, seed):
    rng = np.random.RandomState(seed)
    vocab = spa.Vocabulary(32, rng=rng)
    vocab.populate('A; B')

    with spa.Network(seed=seed, vocabs=VocabularyMap([vocab])) as model:
        model.bind = spa.Bind(vocab=32)

        def inputA(t):
            if 0 <= t < 0.1:
                return 'A'
            else:
                return 'B'

        model.input = spa.Transcode(inputA, output_vocab=vocab)
        model.input >> model.bind.input_a
        spa.sym.A >> model.bind.input_b

    with model:
        p = nengo.Probe(model.bind.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    error = rmse(vocab.parse("(B*A).normalized()").v, sim.data[p][-1])
    assert error < 0.1

    error = rmse(vocab.parse("(A*A).normalized()").v, sim.data[p][100])
    assert error < 0.1
