import pytest

from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import (
    TAnyVocab,
    TAnyVocabOfDim,
    TScalar,
    TVocabulary,
    Type,
    coerce_types,
)
from nengo_spa.vocabulary import Vocabulary


def test_coercion():
    v1 = TVocabulary(Vocabulary(16))

    assert coerce_types(TAnyVocab, TAnyVocab) is TAnyVocab
    assert coerce_types(TAnyVocab, TScalar) is TAnyVocab
    assert coerce_types(TAnyVocab, v1) == v1
    assert coerce_types(TScalar, TScalar) == TScalar
    assert coerce_types(TScalar, TScalar, TScalar) == TScalar
    assert coerce_types(TScalar, v1) == v1
    assert coerce_types(v1, v1) == v1
    assert coerce_types(TAnyVocab, v1, TScalar) == v1
    assert coerce_types(TScalar, TScalar, v1, TScalar, v1) == v1


def test_coercion_errors():
    with pytest.raises(SpaTypeError) as err:
        coerce_types(TVocabulary(Vocabulary(16)), TVocabulary(Vocabulary(16)))
    assert "Different vocabularies" in str(err.value)

    with pytest.raises(SpaTypeError) as err:
        coerce_types(TAnyVocabOfDim(16), TAnyVocabOfDim(32))
    assert "Dimensionality mismatch" in str(err.value)

    with pytest.raises(SpaTypeError) as err:
        coerce_types(Type("x"), Type("y"))
    assert "Incompatible types" in str(err.value)


def test_non_equality():
    v = Vocabulary(16)
    tv1 = TVocabulary(v)
    tv2 = TVocabulary(v)
    tvx = TVocabulary(Vocabulary(16))
    assert tv1 == tv2
    assert not (tv1 != tv2)
    assert tv1 != tvx
    assert not (tv1 == tvx)
