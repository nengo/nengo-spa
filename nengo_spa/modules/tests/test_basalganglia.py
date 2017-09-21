import numpy as np

import nengo
import nengo_spa as spa


def test_basic(Simulator, plt, seed):
    with nengo.Network(seed=seed) as model:
        bg = spa.BasalGanglia(action_count=5, seed=seed)
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)
        p = nengo.Probe(bg.output, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > -0.1
    assert np.all(output[1:] < -0.8)


def test_basal_ganglia(Simulator, seed, plt):
    with spa.Network(seed=seed) as m:
        m.vision = spa.State(vocab=16)
        m.motor = spa.State(vocab=16)
        m.compare = spa.Compare(vocab=16)

        def input(t):
            if t < 0.1:
                return '0'
            elif t < 0.2:
                return 'CAT'
            elif t < 0.3:
                return 'DOG*~CAT'
            elif t < 0.4:
                return 'PARROT'
            elif t < 0.5:
                return 'MOUSE'
            else:
                return '0'
        m.encode = spa.Transcode(input, output_vocab=16)

        # test all acceptable condition formats
        actions = spa.Actions('''
            ifmax 0.5: A -> m.motor
            elifmax dot(m.vision, CAT): B -> m.motor
            elifmax dot(m.vision * CAT, DOG): C -> m.motor
            elifmax 2 * dot(m.vision, CAT * 0.5): D -> m.motor
            elifmax dot(m.vision, CAT) + 0.5 - dot(m.vision, CAT): E -> m.motor
            elifmax dot(m.vision, PARROT) + m.compare: F -> m.motor
            elifmax 0.5 * dot(m.vision, MOUSE) + 0.5 * m.compare: G -> m.motor
            elifmax (dot(m.vision, MOUSE) - m.compare) * 0.5: H -> m.motor

            always:
                m.encode -> m.vision
                SHOOP -> m.compare.input_a
                SHOOP -> m.compare.input_b
        ''')
        bg = actions[0].bg

        p = nengo.Probe(bg.input, 'output', synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)
    t = sim.trange()

    plt.plot(t, sim.data[p])
    plt.legend(["A", "B", "C", "D", "E", "F", "G", "H"])
    plt.title('Basal Ganglia output')

    # assert the basal ganglia is prioritizing things correctly
    # Motor F
    assert sim.data[p][t == 0.4, 5] > 0.8
    # Motor G
    assert sim.data[p][t == 0.5, 6] > 0.8
    # Motor A
    assert 0.6 > sim.data[p][t == 0.1, 0] > 0.4
    # Motor B
    assert sim.data[p][t == 0.2, 1] > 0.8
    # Motor C
    assert sim.data[p][t == 0.3, 2] > 0.5

    # Motor B should be the same as Motor D
    assert np.allclose(sim.data[p][:, 1], sim.data[p][:, 3])
    # Motor A should be the same as Motor E
    assert np.allclose(sim.data[p][:, 0], sim.data[p][:, 4])


def test_scalar_product():
    with spa.Network() as model:
        model.scalar = spa.Scalar()
        spa.Actions('ifmax model.scalar * model.scalar: 1 -> model.scalar')
    # just testing network construction without exception here


def test_constructed_input_connections_are_accessible():
    with spa.Network() as model:
        model.config[spa.State].vocab = 16
        model.state1 = spa.State()
        model.state2 = spa.State()

        actions = spa.Actions('ifmax dot(model.state1, A): A -> model.state2')
        bg = actions[0].bg

        assert isinstance(bg.input_connections[0], nengo.Connection)
