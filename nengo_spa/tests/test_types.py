import pytest

from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import coerce_types, TAnyVocab, TScalar, TVocabulary
from nengo_spa.vocab import Vocabulary


def test_coercion():
    v1 = TVocabulary(Vocabulary(16))
    v2 = TVocabulary(Vocabulary(16))

    assert coerce_types(TAnyVocab, TAnyVocab) is TAnyVocab
    assert coerce_types(TAnyVocab, TScalar) is TAnyVocab
    assert coerce_types(TAnyVocab, v1) == v1
    assert coerce_types(TScalar, TScalar) == TScalar
    assert coerce_types(TScalar, TScalar, TScalar) == TScalar
    assert coerce_types(TScalar, v1) == v1
    assert coerce_types(v1, v1) == v1
    assert coerce_types(TAnyVocab, v1, TScalar) == v1
    assert coerce_types(TScalar, TScalar, v1, TScalar, v1) == v1
    with pytest.raises(SpaTypeError):
        coerce_types(v1, v2)
