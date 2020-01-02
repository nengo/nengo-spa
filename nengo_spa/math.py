"""Functions to evaluate analytically derived equations related to the SPA."""

from nengo.dists import CosineSimilarity


def prob_cleanup(similarity, dimensions, vocab_size):
    """Estimate the chance of successful cleanup.

    This returns the chance that, out of *vocab_size* randomly chosen
    vectors, none of them will be closer to a particular
    vector than the value given by *similarity*. To use this, compare
    your noisy vector with the ideal vector, pass that value in as
    the similarity parameter, and set *vocab_size* to be the number of
    competing vectors.

    Requires SciPy.
    """
    p = CosineSimilarity(dimensions).cdf(similarity)
    if similarity < 1.0 and p == 1.0:
        raise ArithmeticError("Insufficient floating point precision to compute value.")
    return p ** vocab_size
