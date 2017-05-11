import numpy as np

import nengo
import nengo_spa as spa


def test_basic():
    with spa.Network() as model:
        model.state = spa.State(vocab=16)

    input = model.get_network_input('state')
    output = model.get_network_output('state')
    assert input[0] is model.state.input
    assert output[0] is model.state.output
    assert input[1] is output[1]
    assert input[1].dimensions == 16


def test_neurons():
    with spa.Network() as model:
        model.state = spa.State(
            vocab=16, neurons_per_dimension=2, represent_identity=False)

    assert len(model.state.state_ensembles.ensembles) == 1
    assert model.state.state_ensembles.ensembles[0].n_neurons == 16 * 2

    with spa.Network() as model:
        model.state = spa.State(
            vocab=16, subdimensions=1, neurons_per_dimension=2,
            represent_identity=False)

    assert len(model.state.state_ensembles.ensembles) == 16
    assert model.state.state_ensembles.ensembles[0].n_neurons == 2


def test_no_feedback_run(Simulator, plt, seed):
    with spa.Network(seed=seed) as model:
        model.state = spa.State(vocab=32, feedback=0.0)

        def state_input(t):
            if 0 <= t < 0.3:
                return 'A'
            elif 0.2 <= t < 0.6:
                return 'B'
            else:
                return '0'
        model.state_input = spa.Input()
        model.state_input.state = state_input

    state, vocab = model.get_network_output('state')

    with model:
        p = nengo.Probe(state, 'output', synapse=0.05)

    with Simulator(model) as sim:
        sim.run(0.8)

    data = np.dot(sim.data[p], vocab.vectors.T)
    plt.plot(sim.trange(), data)
    assert data[299, 0] > 0.9
    assert data[299, 1] < 0.2
    assert data[599, 0] < 0.2
    assert data[599, 1] > 0.9
    assert data[799, 0] < 0.2
    assert data[799, 1] < 0.2


def test_memory_run(Simulator, seed, plt):
    with spa.Network(seed=seed) as model:
        model.memory = spa.State(vocab=32, feedback=1.0,
                                 feedback_synapse=0.01)

        def state_input(t):
            if 0 <= t < 0.05:
                return 'A'
            else:
                return '0'

        model.state_input = spa.Input()
        model.state_input.memory = state_input

    memory, vocab = model.get_network_output('memory')

    with model:
        p = nengo.Probe(memory, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)
    t = sim.trange()

    similarity = np.dot(sim.data[p], vocab.vectors.T)
    plt.plot(t, similarity)
    plt.ylabel("Similarity to 'A'")
    plt.xlabel("Time (s)")

    # value should peak above 1.0, then decay down to near 1.0
    assert np.mean(similarity[(t > 0.05) & (t < 0.1)]) > 1.0
    assert np.mean(similarity[(t > 0.2) & (t < 0.3)]) > 0.7
    assert np.mean(similarity[t > 0.49]) > 0.5


def test_memory_run_decay(Simulator, plt, seed):
    with spa.Network(seed=seed) as model:
        model.memory = spa.State(vocab=32, feedback=(1.0 - 0.01/0.05),
                                 feedback_synapse=0.01)

        def state_input(t):
            if 0 <= t < 0.05:
                return 'A'
            else:
                return '0'

        model.state_input = spa.Input()
        model.state_input.memory = state_input

    memory, vocab = model.get_network_output('memory')

    with model:
        p = nengo.Probe(memory, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)
    data = np.dot(sim.data[p], vocab.vectors.T)

    t = sim.trange()
    plt.plot(t, data)

    assert data[t == 0.05, 0] > 1.0
    assert data[t == 0.299, 0] < 0.4


def test_represent_identity(Simulator, seed):
    d = 32
    with spa.Network(seed=seed) as model:
        model.memory = spa.State(d, represent_identity=True)
        model.input = spa.Input()
        model.input.memory = '1'
        p = nengo.Probe(model.memory.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert np.all(sim.data[p][sim.trange() > 0.2][:, 0] > 0.9)
    assert np.all(np.abs(sim.data[p][sim.trange() > 0.2][:, 1:]) < 0.1)
