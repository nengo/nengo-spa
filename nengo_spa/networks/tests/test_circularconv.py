import nengo
import numpy as np
import pytest
from nengo.utils.numpy import rms

import nengo_spa
from nengo_spa.algebras.hrr_algebra import HrrAlgebra
from nengo_spa.networks.circularconvolution import transform_in, transform_out


@pytest.mark.parametrize("invert_a", [True, False])
@pytest.mark.parametrize("invert_b", [True, False])
def test_circularconv_transforms(invert_a, invert_b, rng):
    """Test the circular convolution transforms"""
    dims = 100
    x = a = rng.randn(dims)
    y = b = rng.randn(dims)
    inv = HrrAlgebra().get_inversion_matrix(dims)
    if invert_a:
        a = np.dot(inv, a)
    if invert_b:
        b = np.dot(inv, b)
    z0 = HrrAlgebra().bind(a, b)

    tr_a = transform_in(dims, "A", invert_a)
    tr_b = transform_in(dims, "B", invert_b)
    tr_out = transform_out(dims)
    XY = np.dot(tr_a, x) * np.dot(tr_b, y)
    z1 = np.dot(tr_out, XY)

    assert np.allclose(z0, z1)


def test_input_magnitude(Simulator, seed, rng, dims=16, magnitude=10):
    """Test to make sure the magnitude scaling works.

    Builds two different CircularConvolution networks, one with the correct
    magnitude and one with 1.0 as the input_magnitude.
    """
    neurons_per_product = 128

    a = rng.normal(scale=np.sqrt(1.0 / dims), size=dims) * magnitude
    b = rng.normal(scale=np.sqrt(1.0 / dims), size=dims) * magnitude
    result = HrrAlgebra().bind(a, b)

    model = nengo.Network(label="circular conv", seed=seed)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with model:
        input_a = nengo.Node(a)
        input_b = nengo.Node(b)
        cconv = nengo_spa.networks.CircularConvolution(
            neurons_per_product, dimensions=dims, input_magnitude=magnitude
        )
        nengo.Connection(input_a, cconv.input_a, synapse=None)
        nengo.Connection(input_b, cconv.input_b, synapse=None)
        res_p = nengo.Probe(cconv.output)
        cconv_bad = nengo_spa.networks.CircularConvolution(
            neurons_per_product, dimensions=dims, input_magnitude=1
        )  # incorrect magnitude
        nengo.Connection(input_a, cconv_bad.input_a, synapse=None)
        nengo.Connection(input_b, cconv_bad.input_b, synapse=None)
        res_p_bad = nengo.Probe(cconv_bad.output)
    with Simulator(model) as sim:
        sim.run(0.01)

    error = rms(result - sim.data[res_p][-1]) / (magnitude ** 2)
    error_bad = rms(result - sim.data[res_p_bad][-1]) / (magnitude ** 2)

    assert error < 0.2
    assert error_bad > 0.1


@pytest.mark.parametrize("dims", [8, 32])
def test_neural_accuracy(Simulator, seed, rng, dims, neurons_per_product=128):
    a = rng.normal(scale=np.sqrt(1.0 / dims), size=dims)
    b = rng.normal(scale=np.sqrt(1.0 / dims), size=dims)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    result = HrrAlgebra().bind(a, b)

    model = nengo.Network(label="circular conv", seed=seed)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with model:
        input_a = nengo.Node(a)
        input_b = nengo.Node(b)
        cconv = nengo_spa.networks.CircularConvolution(
            neurons_per_product, dimensions=dims
        )
        nengo.Connection(input_a, cconv.input_a, synapse=None)
        nengo.Connection(input_b, cconv.input_b, synapse=None)
        res_p = nengo.Probe(cconv.output)
    with Simulator(model) as sim:
        sim.run(0.01)

    error = rms(result - sim.data[res_p][-1])

    assert error < 0.2
