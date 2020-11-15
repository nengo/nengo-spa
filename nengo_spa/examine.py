"""Functions to examine data and vectors produced by SPA."""

from itertools import combinations

import nengo.utils.numpy as npext
import numpy as np
from nengo.exceptions import ValidationError

from nengo_spa.semantic_pointer import SemanticPointer
from nengo_spa.typechecks import is_iterable
from nengo_spa.vocabulary import Vocabulary


def similarity(data, vocab, normalize=False):
    """Return the similarity between simulation data and Semantic Pointers.

    Computes the dot products between all Semantic Pointers in the Vocabulary
    and the simulation data for each timestep. If ``normalize=True``,
    normalizes all vectors to compute the cosine similarity.

    Parameters
    ----------
    data: (D,) or (T, D) array_like
        The *D*-dimensional data for *T* timesteps used for comparison.
    vocab: Vocabulary or array_like
        Vocabulary (or list of vectors) used to calculate the similarity
        values.
    normalize : bool, optional
        Whether to normalize all vectors, to compute the cosine similarity.
    """

    if isinstance(data, SemanticPointer):
        data = data.v

    if isinstance(vocab, Vocabulary):
        vectors = vocab.vectors
    elif is_iterable(vocab):
        if isinstance(next(iter(vocab)), SemanticPointer):
            vocab = [p.v for p in vocab]
        vectors = np.array(vocab, copy=False, ndmin=2)
    else:
        raise ValidationError(
            "%r object is not a valid vocabulary" % (type(vocab).__name__), attr="vocab"
        )

    dots = np.dot(vectors, data.T)

    if normalize:
        # Zero-norm vectors should return zero, so avoid divide-by-zero error
        eps = np.nextafter(0, 1)  # smallest float above zero
        dnorm = np.maximum(npext.norm(data.T, axis=0, keepdims=True), eps)
        vnorm = np.maximum(npext.norm(vectors, axis=1, keepdims=True), eps)

        if len(dots.shape) == 1:
            vnorm = np.squeeze(vnorm)

        dots /= dnorm
        dots /= vnorm

    return dots.T


def pairs(vocab):
    """Return expressions for all possible combinations to bind *vocab*'s keys.

    Examples
    --------

    >>> vocab = nengo_spa.Vocabulary(32)
    >>> vocab.populate('A; B; C')
    >>> sorted(nengo_spa.pairs(vocab))
    ['A*B', 'A*C', 'B*C']
    """

    return set(x + "*" + y for x, y in combinations(vocab.keys(), 2))


def text(
    v,
    vocab,
    minimum_count=1,
    maximum_count=None,
    threshold=0.1,
    join=";",
    terms=None,
    normalize=False,
):
    """Return a human-readable text version of the provided vector.

    This is meant to give a quick text version of a vector for display
    purposes. To do this, compute the dot product between the vector
    and all the terms in the vocabulary. The top few vectors are
    chosen for inclusion in the text. It will try to only return
    terms with a match above the *threshold*, but will always return
    at least *minimum_count* and at most maximum_count terms. Terms
    are sorted from most to least similar.

    Parameters
    ----------
    v : SemanticPointer or array_like
        The vector to convert into text.
    minimum_count : int, optional
        Always return at least this many terms in the text.
    maximum_count : int, optional
        Never return more than this many terms in the text.
        If None, all terms will be returned.
    threshold : float, optional
        How small a similarity for a term to be ignored.
    join : str, optional
        The text separator to use between terms.
    terms : list, optional
        Only consider terms in this list of strings.
    normalize : bool
        Whether to normalize the vector before computing similarity.
    """
    if not isinstance(v, SemanticPointer):
        v = SemanticPointer(v)
    if normalize:
        v = v.normalized()

    if terms is None:
        terms = vocab.keys()
        vectors = vocab.vectors
    else:
        vectors = vocab.parse_n(*terms)

    matches = list(zip(similarity(v, vectors), terms))
    matches.sort()
    matches.reverse()

    r = []
    for m in matches:
        if minimum_count is not None and len(r) < minimum_count:
            r.append(m)
        elif maximum_count is not None and len(r) == maximum_count:
            break
        elif threshold is None or m[0] > threshold:
            r.append(m)
        else:
            break

    return join.join(["%0.2f%s" % (sim, key) for (sim, key) in r])
