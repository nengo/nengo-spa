import nengo
import numpy as np
import pytest
from nengo.exceptions import ValidationError

import nengo_spa as spa
from nengo_spa import Vocabulary
from nengo_spa.examine import similarity
from nengo_spa.modules.associative_memory import (
    IAAssocMem,
    ThresholdingAssocMem,
    WTAAssocMem,
)
from nengo_spa.testing import assert_sp_close

filtered_step_fn = lambda x: np.maximum(1.0 - np.exp(-15.0 * x), 0.0)


def test_am_basic(Simulator, plt, seed, rng):
    """Basic associative memory test."""

    d = 64
    vocab = Vocabulary(d, pointer_gen=rng)
    vocab.populate("A; B; C; D")

    with spa.Network("model", seed=seed) as m:
        m.am = ThresholdingAssocMem(
            threshold=0.3,
            input_vocab=vocab,
            mapping=vocab.keys(),
            function=filtered_step_fn,
        )
        spa.sym.A >> m.am

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.2)
    t = sim.trange()

    plt.subplot(3, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.subplot(3, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab))
    plt.plot(t[t > 0.15], np.ones(t.shape)[t > 0.15] * 0.95, c="g", lw=2)
    plt.ylabel("Output")

    assert_sp_close(t, sim.data[in_p], vocab["A"], skip=0.15, atol=0.05)
    assert_sp_close(t, sim.data[out_p], vocab["A"], skip=0.15)


def test_am_threshold(Simulator, plt, seed, rng):
    """Associative memory thresholding with differing input/output vocabs."""
    d = 64
    vocab = Vocabulary(d, pointer_gen=rng)
    vocab.populate("A; B; C; D")

    d2 = int(d / 2)
    vocab2 = Vocabulary(d2, pointer_gen=rng)
    vocab2.populate("A; B; C; D")

    def input_func(t):
        return "0.49 * A" if t < 0.1 else "0.8 * B"

    with spa.Network("model", seed=seed) as m:
        m.am = ThresholdingAssocMem(
            threshold=0.5,
            input_vocab=vocab,
            output_vocab=vocab2,
            function=filtered_step_fn,
            mapping="by-key",
        )
        m.stimulus = spa.Transcode(input_func, output_vocab=vocab)
        m.stimulus >> m.am

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.3)
    t = sim.trange()
    below_th = t < 0.1
    above_th = t > 0.25

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.subplot(2, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab2))
    plt.plot(t[above_th], np.ones(t.shape)[above_th] * 0.9, c="g", lw=2)
    plt.ylabel("Output")

    assert np.mean(sim.data[out_p][below_th]) < 0.01
    assert_sp_close(t, sim.data[out_p], vocab2["B"], skip=0.25, duration=0.05)


def test_am_wta(Simulator, plt, seed, rng):
    """Test the winner-take-all ability of the associative memory."""

    d = 64
    vocab = Vocabulary(d, pointer_gen=rng)
    vocab.populate("A; B; C; D")

    def input_func(t):
        if t < 0.2:
            return "A + 0.8 * B"
        elif t < 0.3:
            return "0"
        else:
            return "0.8 * A + B"

    with spa.Network("model", seed=seed) as m:
        m.am = WTAAssocMem(
            threshold=0.3,
            input_vocab=vocab,
            mapping=vocab.keys(),
            function=filtered_step_fn,
        )
        m.stimulus = spa.Transcode(input_func, output_vocab=vocab)
        m.stimulus >> m.am

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)
    t = sim.trange()
    more_a = (t > 0.15) & (t < 0.2)
    more_b = t > 0.45

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.subplot(2, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab))
    plt.plot(t[more_a], np.ones(t.shape)[more_a] * 0.9, c="g", lw=2)
    plt.plot(t[more_b], np.ones(t.shape)[more_b] * 0.9, c="g", lw=2)
    plt.ylabel("Output")

    assert_sp_close(t, sim.data[out_p], vocab["A"], skip=0.15, duration=0.05)
    assert_sp_close(t, sim.data[out_p], vocab["B"], skip=0.45, duration=0.05)


def test_am_ia(Simulator, plt, seed, rng):
    """Test the winner-take-all ability of the IA memory."""

    d = 64
    vocab = Vocabulary(d, pointer_gen=rng)
    vocab.populate("A; B; C; D")

    def input_func(t):
        if t < 0.2:
            return "A + 0.8 * B"
        else:
            return "0.6 * A + B"

    with spa.Network("model", seed=seed) as m:
        m.am = IAAssocMem(input_vocab=vocab, mapping=vocab.keys())
        m.stimulus = spa.Transcode(input_func, output_vocab=vocab)
        m.reset = nengo.Node(lambda t: 0.2 < t < 0.4)

        m.stimulus >> m.am
        nengo.Connection(m.reset, m.am.input_reset, synapse=0.1)

        in_p = nengo.Probe(m.am.input)
        reset_p = nengo.Probe(m.reset)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.7)

    t = sim.trange()
    more_a = (t > 0.15) & (t < 0.2)
    more_b = t > 0.65

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.plot(t, sim.data[reset_p], c="k", linestyle="--")
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.subplot(2, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab))
    plt.plot(t[more_a], np.ones(t.shape)[more_a] * 0.9, c="tab:blue", lw=2)
    plt.plot(t[more_b], np.ones(t.shape)[more_b] * 0.9, c="tab:orange", lw=2)
    plt.ylabel("Output")

    assert_sp_close(t, sim.data[out_p], vocab["A"], skip=0.15, duration=0.05)
    assert_sp_close(t, sim.data[out_p], vocab["B"], skip=0.65, duration=0.05)


def test_am_default_output(Simulator, plt, seed, rng):
    d = 64
    vocab = Vocabulary(d, pointer_gen=rng)
    vocab.populate("A; B; C; D")

    def input_func(t):
        return "0.2 * A" if t < 0.25 else "A"

    with spa.Network("model", seed=seed) as m:
        m.am = ThresholdingAssocMem(
            threshold=0.5,
            input_vocab=vocab,
            mapping=vocab.keys(),
            function=filtered_step_fn,
        )
        m.am.add_default_output("D", 0.5)
        m.stimulus = spa.Transcode(input_func, output_vocab=vocab)
        m.stimulus >> m.am

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)

    t = sim.trange()
    below_th = (t > 0.15) & (t < 0.25)
    above_th = t > 0.4

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.subplot(2, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab))
    plt.plot(t[below_th], np.ones(t.shape)[below_th] * 0.9, c="c", lw=2)
    plt.plot(t[above_th], np.ones(t.shape)[above_th] * 0.9, c="b", lw=2)
    plt.plot(t[above_th], np.ones(t.shape)[above_th] * 0.1, c="c", lw=2)
    plt.ylabel("Output")

    assert np.all(similarity(sim.data[out_p][below_th], [vocab["D"].v]) > 0.9)
    assert np.all(similarity(sim.data[out_p][above_th], [vocab["D"].v]) < 0.15)
    assert np.all(similarity(sim.data[out_p][above_th], [vocab["A"].v]) > 0.9)


def test_am_spa_keys_as_expressions(Simulator, plt, seed, rng):
    """Provide semantic pointer expressions as input and output keys."""
    d = 64

    vocab_in = Vocabulary(d, pointer_gen=rng)
    vocab_out = Vocabulary(d, pointer_gen=rng)

    vocab_in.populate("A; B")
    vocab_out.populate("C; D")

    in_keys = ["A", "A*B"]
    out_keys = ["C*D", "C+D"]
    mapping = dict(zip(in_keys, out_keys))

    with spa.Network(seed=seed) as m:
        m.am = ThresholdingAssocMem(
            threshold=0.3, input_vocab=vocab_in, output_vocab=vocab_out, mapping=mapping
        )

        m.inp = spa.Transcode(
            lambda t: "A" if t < 0.1 else "A*B", output_vocab=vocab_in
        )
        m.inp >> m.am

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.2)

    # Specify t ranges
    t = sim.trange()
    t_item1 = (t > 0.075) & (t < 0.1)
    t_item2 = (t > 0.175) & (t < 0.2)

    # Modify vocabularies (for plotting purposes)
    vocab_in.add("AxB", vocab_in.parse(in_keys[1]).v)
    vocab_out.add("CxD", vocab_out.parse(out_keys[0]).v)

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab_in))
    plt.ylabel("Input: " + ", ".join(in_keys))
    plt.legend(vocab_in.keys(), loc="best")
    plt.ylim(top=1.1)
    plt.subplot(2, 1, 2)
    for t_item, c, k in zip([t_item1, t_item2], ["b", "g"], out_keys):
        plt.plot(
            t,
            similarity(sim.data[out_p], [vocab_out.parse(k).v], normalize=True),
            label=k,
            c=c,
        )
        plt.plot(t[t_item], np.ones(t.shape)[t_item] * 0.9, c=c, lw=2)
    plt.ylabel("Output: " + ", ".join(out_keys))
    plt.legend(loc="best")

    assert (
        np.mean(
            similarity(
                sim.data[out_p][t_item1], vocab_out.parse(out_keys[0]).v, normalize=True
            )
        )
        > 0.9
    )
    assert (
        np.mean(
            similarity(
                sim.data[out_p][t_item2], vocab_out.parse(out_keys[1]).v, normalize=True
            )
        )
        > 0.9
    )


def test_invalid_mapping_string():
    with spa.Network():
        with pytest.raises(ValidationError):
            ThresholdingAssocMem(
                threshold=0.3, input_vocab=16, output_vocab=32, mapping="invalid"
            )


@pytest.mark.parametrize(
    "cls_and_args",
    ((ThresholdingAssocMem, (0.3,)), (WTAAssocMem, (0.3,)), (IAAssocMem, ())),
)
def test_int_vocabs(cls_and_args):
    cls, args = cls_and_args
    with spa.Network() as model:
        model.vocabs.get_or_create(32).populate("A")
        # no assertion, just ensure no exception
        cls(*args, input_vocab=32, mapping=["A"])


@pytest.mark.parametrize(
    "cls_and_args",
    ((ThresholdingAssocMem, (0.3,)), (WTAAssocMem, (0.3,)), (IAAssocMem, ())),
)
def test_enforces_explicit_mapping(cls_and_args):
    cls, args = cls_and_args
    with spa.Network() as model:
        model.vocabs.get_or_create(32).populate("A")
        with pytest.raises(TypeError):
            cls(*args, input_vocab=32)


@pytest.mark.parametrize(
    "cls_and_args",
    ((ThresholdingAssocMem, (0.3,)), (WTAAssocMem, (0.3,)), (IAAssocMem, ())),
)
def test_enforces_at_least_one_item(cls_and_args):
    cls, args = cls_and_args
    with spa.Network() as model:
        model.vocabs.get_or_create(32).populate("A")
        with pytest.raises(ValidationError, match="At least one"):
            cls(*args, input_vocab=32, mapping=[])
