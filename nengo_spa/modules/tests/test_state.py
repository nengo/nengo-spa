import nengo
import numpy as np

import nengo_spa as spa


def test_neurons():
    with spa.Network():
        state = spa.State(
            vocab=16, neurons_per_dimension=2, represent_cc_identity=False
        )

    assert len(state.state_ensembles.ensembles) == 1
    assert state.state_ensembles.ensembles[0].n_neurons == 16 * 2

    with spa.Network():
        state = spa.State(
            vocab=16,
            subdimensions=1,
            neurons_per_dimension=2,
            represent_cc_identity=False,
        )

    assert len(state.state_ensembles.ensembles) == 16
    assert state.state_ensembles.ensembles[0].n_neurons == 2


def test_no_feedback_run(Simulator, plt, seed):
    with spa.Network(seed=seed) as model:
        state = spa.State(vocab=32, feedback=0.0)

        def state_input(t):
            if 0 <= t < 0.3:
                return "A"
            elif 0.2 <= t < 0.6:
                return "B"
            else:
                return "0"

        state_input = spa.Transcode(state_input, output_vocab=32)
        state_input >> state

        p = nengo.Probe(state.output, synapse=0.05)

    with Simulator(model) as sim:
        sim.run(0.8)

    data = np.dot(sim.data[p], state.vocab.vectors.T)
    plt.plot(sim.trange(), data)
    assert data[299, 0] > 0.9
    assert data[299, 1] < 0.2
    assert data[599, 0] < 0.2
    assert data[599, 1] > 0.9
    assert data[799, 0] < 0.2
    assert data[799, 1] < 0.2


def test_memory_run(Simulator, seed, plt):
    with spa.Network(seed=seed) as model:
        memory = spa.State(vocab=32, feedback=1.0, feedback_synapse=0.01)

        def state_input(t):
            if 0 <= t < 0.05:
                return "A"
            else:
                return "0"

        state_input = spa.Transcode(state_input, output_vocab=32)
        state_input >> memory

        p = nengo.Probe(memory.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)
    t = sim.trange()

    similarity = np.dot(sim.data[p], memory.vocab.vectors.T)
    plt.plot(t, similarity)
    plt.ylabel("Similarity to 'A'")
    plt.xlabel("Time (s)")

    # value should peak above 1.0, then decay down to near 1.0
    assert np.mean(similarity[(t > 0.05) & (t < 0.1)]) > 1.0
    assert np.mean(similarity[(t > 0.2) & (t < 0.3)]) > 0.7
    assert np.mean(similarity[t > 0.49]) > 0.5


def test_memory_run_decay(Simulator, plt, seed):
    with spa.Network(seed=seed) as model:
        memory = spa.State(
            vocab=32, feedback=(1.0 - 0.01 / 0.05), feedback_synapse=0.01
        )

        def state_input(t):
            if 0 <= t < 0.05:
                return "A"
            else:
                return "0"

        state_input = spa.Transcode(state_input, output_vocab=32)
        state_input >> memory

        p = nengo.Probe(memory.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)
    data = np.dot(sim.data[p], memory.vocab.vectors.T)

    t = sim.trange()
    plt.plot(t, data)

    assert data[t == 0.05, 0] > 1.0
    assert data[t == 0.299, 0] < 0.4


def test_represent_cc_identity(Simulator, seed):
    d = 32
    with spa.Network(seed=seed) as model:
        memory = spa.State(d, represent_cc_identity=True)
        spa.semantic_pointer.Identity(d) >> memory
        p = nengo.Probe(memory.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert np.all(sim.data[p][sim.trange() > 0.2][:, 0] > 0.9)
    assert np.all(np.abs(sim.data[p][sim.trange() > 0.2][:, 1:]) < 0.1)
