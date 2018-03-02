# -*- coding: utf-8 -*-

from nengo.exceptions import NengoWarning, ValidationError
import nengo.solvers
import numpy as np
from numpy.testing import assert_equal
import pytest

from nengo_spa import Vocabulary
from nengo_spa.exceptions import SpaParseError
from nengo_spa.pointer import Identity, SemanticPointer
from nengo_spa.vocab import (
    VocabularyMap, VocabularyMapParam, VocabularyOrDimParam)


def test_add(rng):
    v = Vocabulary(3, rng=rng)
    v.add('A', [1, 2, 3])
    v.add('B', [4, 5, 6])
    v.add('C', [7, 8, 9])
    assert np.allclose(v.vectors, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_populate(rng):
    v = Vocabulary(64, rng=rng)

    v.populate('')
    v.populate(' \r\n\t')
    assert len(v) == 0

    v.populate('A')
    assert 'A' in v

    v.populate('B; C')
    assert 'B' in v
    assert 'C' in v

    v.populate('D.unitary()')
    assert 'D' in v
    np.testing.assert_almost_equal(np.linalg.norm(v['D'].v), 1.)
    np.testing.assert_almost_equal(np.linalg.norm((v['D'] * v['D']).v), 1.)

    v.populate('E = A + 2 * B')
    assert np.allclose(v['E'].v, v.parse('A + 2 * B').v)
    assert np.linalg.norm(v['E'].v) > 2.

    v.populate('F = (A + 2 * B).normalized()')
    assert np.allclose(v['F'].v, v.parse('A + 2 * B').normalized().v)
    np.testing.assert_almost_equal(np.linalg.norm(v['F'].v), 1.)

    v.populate('G = A; H')
    assert np.allclose(v['G'].v, v['A'].v)
    assert 'H' in v

    # Assigning non-existing pointer
    with pytest.raises(NameError):
        v.populate('I = J')

    # Redefining
    with pytest.raises(ValidationError):
        v.populate('H = A')

    # Calling non existing function
    with pytest.raises(AttributeError):
        v.populate('I = H.invalid()')

    # invalid names: lowercase, unicode
    with pytest.raises(SpaParseError):
        v.populate('x = A')
    # with pytest.raises(SpaParseError):
    v.populate(u'Aα = A')


def test_populate_with_transform_on_first_vector(rng):
    v = Vocabulary(64, rng=rng)

    v.populate('A.unitary()')
    assert 'A' in v
    assert np.allclose(v['A'].v, v['A'].unitary().v)


def test_populate_with_transform_on_nonstrict_vocab(rng):
    v = Vocabulary(64, rng=rng, strict=False)

    v.populate('A.unitary()')
    assert 'A' in v
    assert np.allclose(v['A'].v, v['A'].unitary().v)


def test_parse(rng):
    v = Vocabulary(64, rng=rng)
    v.populate('A; B; C')
    A = v.parse('A')
    B = v.parse('B')
    C = v.parse('C')
    assert np.allclose((A * B).v, v.parse('A * B').v)
    assert np.allclose((A * ~B).v, v.parse('A * ~B').v)
    assert np.allclose((A + B).v, v.parse('A + B').v)
    assert np.allclose((A - (B*C)*3 + ~C).v, v.parse('A-(B*C)*3+~C').v)

    assert np.allclose(v.parse('0').v, np.zeros(64))
    assert np.allclose(v.parse('1').v, np.eye(64)[0])
    assert np.allclose(v.parse('1.7').v, np.eye(64)[0] * 1.7)

    with pytest.raises(SyntaxError):
        v.parse('A((')
    with pytest.raises(SpaParseError):
        v.parse('"hello"')
    with pytest.raises(SpaParseError):
        v.parse('"hello"')


def test_parse_n(rng):
    v = Vocabulary(64, rng=rng)
    v.populate('A; B; C')
    A = v.parse('A')
    B = v.parse('B')

    parsed = v.parse_n('A', 'A*B', 'A+B', '3')
    assert np.allclose(parsed[0].v, A.v)
    assert np.allclose(parsed[1].v, (A * B).v)
    assert np.allclose(parsed[2].v, (A + B).v)
    assert np.allclose(parsed[3].v, 3 * Identity(64).v)


def test_invalid_dimensions():
    with pytest.raises(ValidationError):
        Vocabulary(1.5)
    with pytest.raises(ValidationError):
        Vocabulary(0)
    with pytest.raises(ValidationError):
        Vocabulary(-1)


def test_capital(rng):
    v = Vocabulary(16, rng=rng)
    with pytest.raises(SpaParseError):
        v.parse('a')
    with pytest.raises(SpaParseError):
        v.parse('A+B+C+a')


@pytest.mark.parametrize('solver', [None, nengo.solvers.Lstsq()])
def test_transform(recwarn, rng, solver):
    v1 = Vocabulary(32, strict=False, rng=rng)
    v2 = Vocabulary(64, strict=False, rng=rng)
    v1.populate('A; B; C')
    v2.populate('A; B; C')
    A = v1['A']
    B = v1['B']
    C = v1['C']

    # Test transform from v1 to v2 (full vocbulary)
    # Expected: np.dot(t, A.v) ~= v2.parse('A')
    # Expected: np.dot(t, B.v) ~= v2.parse('B')
    # Expected: np.dot(t, C.v) ~= v2.parse('C')
    t = v1.transform_to(v2, solver=solver)

    assert v2.parse('A').compare(np.dot(t, A.v)) > 0.9
    assert v2.parse('C+B').compare(np.dot(t, C.v + B.v)) > 0.9

    # Test transform from v1 to v2 (only 'A' and 'B')
    t = v1.transform_to(v2, keys=['A', 'B'], solver=solver)

    assert v2.parse('A').compare(np.dot(t, A.v)) > 0.9
    assert v2.parse('B').compare(np.dot(t, C.v + B.v)) > 0.9

    # Test warns on missing keys
    v1.populate('D')
    D = v1['D']
    with pytest.warns(NengoWarning):
        v1.transform_to(v2, solver=solver)

    # Test populating missing keys
    t = v1.transform_to(v2, populate=True, solver=solver)
    assert v2.parse('D').compare(np.dot(t, D.v)) > 0.9

    # Test ignores missing keys in source vocab
    v2.populate('E')
    v1.transform_to(v2, populate=True, solver=solver)
    assert 'E' not in v1


def test_create_pointer_warning(rng):
    v = Vocabulary(2, rng=rng)

    # five pointers shouldn't fit
    with pytest.warns(UserWarning):
        v.populate('A; B; C; D; E')


def test_readonly(rng):
    v1 = Vocabulary(32, rng=rng)
    v1.populate('A;B;C')

    v1.readonly = True

    with pytest.raises(ValueError):
        v1.parse('D')


def test_subset(rng):
    v1 = Vocabulary(32, rng=rng)
    v1.populate('A; B; C; D; E; F; G')

    # Test creating a vocabulary subset
    v2 = v1.create_subset(['A', 'C', 'E'])
    assert list(v2.keys()) == ['A', 'C', 'E']
    assert_equal(v2['A'].v, v1['A'].v)
    assert_equal(v2['C'].v, v1['C'].v)
    assert_equal(v2['E'].v, v1['E'].v)


def test_vocabulary_tracking(rng):
    v = Vocabulary(32, rng=rng)
    v.populate('A')

    assert v['A'].vocab is v
    assert v.parse('2 * A').vocab is v

    v.add('B', SemanticPointer(32))
    v.add('C', SemanticPointer(32, vocab=v))
    with pytest.raises(ValidationError):
        v.add('D', SemanticPointer(32, vocab=Vocabulary(32)))


def test_vocabulary_set(rng):
    v8 = Vocabulary(8)
    v16 = Vocabulary(16)
    v32 = Vocabulary(32)
    vs = VocabularyMap([v8, v16], rng=rng)

    # Behaviour common to set and dict
    assert len(vs) == 2
    assert 8 in vs
    assert 16 in vs
    assert 32 not in vs

    assert v8 in vs
    assert v16 in vs
    assert v32 not in vs
    assert Vocabulary(8) not in vs

    # dict behaviour
    assert vs[8] is v8
    assert vs[16] is v16

    del vs[8]
    assert 8 not in vs

    # set behaviour
    vs.add(v32)
    assert vs[32] is v32
    with pytest.warns(UserWarning):
        vs.add(v32)

    vs.discard(32)
    assert 32 not in vs
    vs.discard(v16)
    assert 16 not in vs

    # creating new vocabs if non existent
    vs.add(v8)
    assert vs.get_or_create(8) is v8
    new = vs.get_or_create(16)
    assert vs[16] is new
    assert new.dimensions == 16
    assert new.rng is rng


def test_vocabulary_map_param():
    class Test(object):
        vocab_map = VocabularyMapParam('vocab_map', readonly=False)

    obj = Test()
    vm = VocabularyMap()
    v16 = Vocabulary(16)
    v32 = Vocabulary(32)

    obj.vocab_map = vm
    assert obj.vocab_map is vm

    obj.vocab_map = [v16, v32]
    assert obj.vocab_map[16] is v16
    assert obj.vocab_map[32] is v32

    with pytest.raises(ValidationError):
        obj.vocab_map = 'incompatible'


def test_vocabulary_or_dim_param():
    v16 = Vocabulary(16)
    v32 = Vocabulary(32)

    class Test(object):
        vocabs = VocabularyMap([v16])
        vocab = VocabularyOrDimParam('vocab', readonly=False)

    obj = Test()

    obj.vocab = v32
    assert obj.vocab is v32

    obj.vocab = 16
    assert obj.vocab is v16

    with pytest.raises(ValidationError):
        obj.vocab = 'incompatible'

    with pytest.raises(ValidationError):
        obj.vocab = 0
