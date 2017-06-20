import nengo
import numpy as np
import pytest

from nengo_spa.networks.identity_ensemble_array import IdentityEnsembleArray
from nengo_spa.pointer import Identity, SemanticPointer


@pytest.mark.parametrize('pointer', [Identity, SemanticPointer])
def test_identity_ensemble_array(Simulator, seed, rng, pointer):
    d = 64
    try:
        v = pointer(d, rng=rng).v
    except TypeError:
        v = pointer(d).v

    with nengo.Network(seed=seed) as model:
        ea = IdentityEnsembleArray(15, d, 4)
        input_v = nengo.Node(v)
        nengo.Connection(input_v, ea.input)
        p = nengo.Probe(ea.output, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.3)

    actual = np.mean(sim.data[p][sim.trange() > 0.2], axis=0)
    assert np.abs(v[0] - actual[0]) < 0.1
    assert np.linalg.norm(v - actual) < 0.2
