import nengo
from nengo.utils.numpy import rms

import nengo_spa as spa
from nengo_spa.vocabulary import VocabularyMap


def test_basic():
    with spa.Network():
        bind = spa.Superposition(2, vocab=16)

    # all inputs and outputs should share the same vocab
    vocab_a = spa.Network.get_input_vocab(bind.inputs[0])
    vocab_b = spa.Network.get_input_vocab(bind.inputs[1])
    assert vocab_a is vocab_b
    assert vocab_a.dimensions == 16
    assert vocab_b.dimensions == 16


def test_run(Simulator, algebra, seed, rng):
    vocab = spa.Vocabulary(32, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B")

    with spa.Network(seed=seed, vocabs=VocabularyMap([vocab])) as model:
        model.superpos = spa.Superposition(2, vocab=32)

        def inputA(t):
            if 0 <= t < 0.1:
                return "A"
            else:
                return "B"

        model.input = spa.Transcode(inputA, output_vocab=vocab)
        model.input >> model.superpos.inputs[0]
        spa.sym.A >> model.superpos.inputs[1]

        p = nengo.Probe(model.superpos.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    error = rms(vocab.parse("(B+A).normalized()").v - sim.data[p][-1])
    assert error < 0.1

    error = rms(vocab.parse("(A+A).normalized()").v - sim.data[p][100])
    assert error < 0.2
