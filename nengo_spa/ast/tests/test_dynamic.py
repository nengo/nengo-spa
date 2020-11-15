import sys

import nengo
import numpy as np
import pytest
from numpy.testing import assert_allclose

import nengo_spa as spa
from nengo_spa.ast.symbolic import PointerSymbol
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.semantic_pointer import SemanticPointer
from nengo_spa.testing import assert_sp_close


def test_scalar_addition(Simulator, seed):
    with spa.Network(seed=seed) as model:
        a = spa.Scalar()
        b = spa.Scalar()

        0.3 >> a
        0.2 >> b
        n1 = (a + b).construct()
        n2 = nengo.Node(size_in=1)
        (a + b).connect_to(n2)

        p1 = nengo.Probe(n1, synapse=0.03)
        p2 = nengo.Probe(n2, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert np.all(np.abs(sim.data[p1][sim.trange() > 0.2] - 0.5) < 0.1)
    assert np.all(np.abs(sim.data[p2][sim.trange() > 0.2] - 0.5) < 0.1)


@pytest.mark.filterwarnings("ignore:.*sidedness:DeprecationWarning")
@pytest.mark.parametrize("op", ["-", "~"])
@pytest.mark.parametrize("suffix", ["", ".output"])
def test_unary_operation_on_module(Simulator, algebra, op, suffix, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A")

    with spa.Network() as model:
        stimulus = spa.Transcode("A", output_vocab=vocab)  # noqa: F841
        x = eval(op + "stimulus" + suffix)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(sim.trange(), sim.data[p], vocab.parse(op + "A"), skip=0.2)


@pytest.mark.parametrize("sidedness", ["l", "r"])
@pytest.mark.parametrize("suffix", ["", ".output"])
def test_inv_operation_on_module(Simulator, algebra, sidedness, suffix, rng):
    try:
        vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
        vocab.populate("A")

        with spa.Network() as model:
            stimulus = spa.Transcode("A", output_vocab=vocab)  # noqa: F841
            x = eval("stimulus" + suffix + "." + sidedness + "inv()")
            p = nengo.Probe(x.construct(), synapse=0.03)

        with Simulator(model) as sim:
            sim.run(0.3)

        assert_sp_close(
            sim.trange(), sim.data[p], vocab.parse("A." + sidedness + "inv()"), skip=0.2
        )
    except NotImplementedError:
        pass


@pytest.mark.parametrize("op", ["+", "-", "*"])
@pytest.mark.parametrize("suffix", ["", ".output"])
def test_binary_operation_on_modules(Simulator, algebra, op, suffix, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)  # noqa: F841
        b = spa.Transcode("B", output_vocab=vocab)  # noqa: F841
        x = eval("a" + suffix + op + "b" + suffix)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("A" + op + "B"), skip=0.2, atol=0.3
    )


@pytest.mark.parametrize("suffix", ["", ".output"])
def test_division_with_fixed(Simulator, suffix, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng)
    vocab.populate("A")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)  # noqa: F841
        x = eval("a" + suffix + "/ 2")
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("0.5 * A"), skip=0.2, atol=0.3
    )


@pytest.mark.parametrize("op", ["+", "-", "*"])
@pytest.mark.parametrize("order", ["AB", "BA"])
def test_binary_operation_on_modules_with_pointer_symbol(
    Simulator, algebra, op, order, seed, rng
):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)  # noqa: F841
        if order == "AB":
            x = eval("a" + op + 'PointerSymbol("B")')
        elif order == "BA":
            x = eval('PointerSymbol("B")' + op + "a")
        else:
            raise ValueError("Invalid order argument.")
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse(order[0] + op + order[1]), skip=0.2
    )


@pytest.mark.parametrize("op", ["+", "-", "*"])
@pytest.mark.parametrize("order", ["AB", "BA"])
def test_binary_operation_on_modules_with_fixed_pointer(
    Simulator, algebra, op, order, seed, rng
):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B")
    b = SemanticPointer(vocab["B"].v)  # noqa: F841

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)  # noqa: F841
        if order == "AB":
            x = eval("a" + op + "b")
        elif order == "BA":
            x = eval("b" + op + "a")
        else:
            raise ValueError("Invalid order argument.")
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(
        sim.trange(),
        sim.data[p],
        vocab.parse(order[0] + op + order[1]),
        skip=0.2,
        atol=0.3,
    )


def test_complex_rule(Simulator, algebra, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B; C; D")

    with spa.Network() as model:
        a = spa.Transcode("A", output_vocab=vocab)
        b = spa.Transcode("B", output_vocab=vocab)

        x = (0.5 * PointerSymbol("C") * a + 0.5 * PointerSymbol("D")) * (
            0.5 * b + a * 0.5
        )
        p = nengo.Probe(x.construct(), synapse=0.3)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(
        sim.trange(),
        sim.data[p],
        vocab.parse("(0.5 * C * A + 0.5 * D) * (0.5 * B + 0.5 * A)"),
        skip=0.2,
        normalized=True,
    )


def test_transformed(Simulator, algebra, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)
        x = PointerSymbol("B") * a
        p = nengo.Probe(x.construct(), synapse=0.3)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(
        sim.trange(), sim.data[p], vocab.parse("B*A"), skip=0.2, normalized=True
    )


def test_transformed_and_pointer_symbol(Simulator, algebra, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)
        x = (a * PointerSymbol("B")) * PointerSymbol("B.rinv()")
        p = nengo.Probe(x.construct(), synapse=0.3)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(
        sim.trange(),
        sim.data[p],
        vocab.parse("A * B * B.rinv()"),
        skip=0.2,
        normalized=True,
    )


def test_transformed_and_network(Simulator, algebra, seed, rng):
    try:
        vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
        vocab.populate("A; B.unitary()")

        with spa.Network(seed=seed) as model:
            a = spa.Transcode("A", output_vocab=vocab)
            b = spa.Transcode("B", output_vocab=vocab)
            x = (a * PointerSymbol("B.linv()")) * b
            p = nengo.Probe(x.construct(), synapse=0.3)

        with Simulator(model) as sim:
            sim.run(0.3)

        assert_sp_close(
            sim.trange(),
            sim.data[p],
            vocab.parse("A * B.linv() * B"),
            skip=0.2,
            normalized=True,
        )
    except NotImplementedError:
        pass


def test_transformed_and_transformed(Simulator, algebra, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng, algebra=algebra)
    vocab.populate("A; B.unitary(); C")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)
        c = spa.Transcode("C", output_vocab=vocab)
        x = (PointerSymbol("B") * a) * (PointerSymbol("B.rinv()") * c)
        p = nengo.Probe(x.construct(), synapse=0.3)

    with Simulator(model) as sim:
        sim.run(0.3)

    assert_sp_close(
        sim.trange(),
        sim.data[p],
        vocab.parse("(B * A) * (B.rinv() * C)"),
        skip=0.2,
        normalized=True,
        atol=0.3,
    )


def test_pointer_symbol_with_dynamic_scalar():
    with spa.Network():
        scalar = spa.Scalar()
        with pytest.raises(SpaTypeError):
            PointerSymbol("A") * scalar


def test_dot(Simulator, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)
        b = spa.Transcode(lambda t: "A" if t <= 0.5 else "B", output_vocab=vocab)
        x = spa.dot(a, b)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    assert_allclose(sim.data[p][(0.3 < t) & (t <= 0.5)], 1.0, atol=0.2)
    assert np.all(sim.data[p][0.8 < t] < 0.2)


@pytest.mark.parametrize("a", ("PointerSymbol('A')", "vocab['A']"))
def test_dot_with_fixed(Simulator, seed, rng, a):
    vocab = spa.Vocabulary(16, pointer_gen=rng)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = eval(a)
        b = spa.Transcode(lambda t: "A" if t <= 0.5 else "B", output_vocab=vocab)
        network_count = len(model.all_networks)
        x = spa.dot(a, b)
        # transform should suffice, no new networks should be created
        assert len(model.all_networks) == network_count
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    assert_allclose(sim.data[p][(0.3 < t) & (t <= 0.5)], 1.0, atol=0.2)
    assert np.all(sim.data[p][0.8 < t] < 0.2)


@pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python 3.5")
def test_dot_matmul(Simulator, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = spa.Transcode("A", output_vocab=vocab)  # noqa: F841
        b = spa.Transcode(  # noqa: F841
            lambda t: "A" if t <= 0.5 else "B", output_vocab=vocab
        )
        x = eval("a @ b")
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    assert_allclose(sim.data[p][(0.3 < t) & (t <= 0.5)], 1.0, atol=0.2)
    assert np.all(sim.data[p][0.8 < t] < 0.2)


@pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python 3.5")
def test_dot_with_fixed_matmul(Simulator, seed, rng):
    vocab = spa.Vocabulary(16, pointer_gen=rng)
    vocab.populate("A; B")

    with spa.Network(seed=seed) as model:
        a = PointerSymbol("A")  # noqa: F841
        b = spa.Transcode(  # noqa: F841
            lambda t: "A" if t <= 0.5 else "B", output_vocab=vocab
        )
        x = eval("a @ b")
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    assert_allclose(sim.data[p][(0.3 < t) & (t <= 0.5)], 1.0, atol=0.2)
    assert np.all(sim.data[p][0.8 < t] < 0.2)


def test_dynamic_translate(Simulator, seed, rng):
    v1 = spa.Vocabulary(64, pointer_gen=rng)
    v1.populate("A; B")
    v2 = spa.Vocabulary(64, pointer_gen=rng)
    v2.populate("A; B")

    with spa.Network(seed=seed) as model:
        source = spa.Transcode("A", output_vocab=v1)
        x = spa.translate(source, v2)
        p = nengo.Probe(x.construct(), synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    assert_sp_close(sim.trange(), sim.data[p], v2["A"], skip=0.3, atol=0.2)
