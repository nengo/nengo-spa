Algebras
--------

Nengo SPA uses elementwise addition for superposition and circular convolution
for binding (`.CircularConvolutionAlgebra`) by default. However, other choices are
viable. In Nengo SPA we call such a specific choice of such operators an
*algebra*. It is easy to change the algebra that is used by Nengo SPA as it is
tied to the vocabulary. To use a different algebra, it suffices to manually
create a vocabulary with the desired algebra and use this in your model::

    import nengo
    import nengo_spa as spa

    vocab = spa.Vocabulary(64, algebra=spa.algebras.VtbAlgebra())

    with spa.Network() as model:
        a = spa.State(vocab)
        b = spa.State(vocab)
        c = spa.State(vocab)
        a * b >> c

In this example the `.VtbAlgebra` (vector-derived transformation binding, VTB)
is used to bind *a* and *b*.

Note that circular convolution is commutative, i.e. :math:`a \circledast
b = b \circledast a`, but this is not true for all algebras. In
particular, the VTB is not commutative. That
means you have to pay attention from which side vectors are bound and unbound.
Moreover, when given :math:`\mathcal{B}(\mathcal{B}(a, b), c)`, it is not
possible to directly unbind :math:`a`, but :math:`c` has to be unbound first
because VTB is not associative.

Custom algebras can be implemented by implementing the `.AbstractAlgebra`
interface. The process involves implementing math versions of the superposition
and binding operator, functions for obtaining specific matrices (such as
inverting a vector), functions for obtaining special elements like the identity
vector, and functions to provide neural implementations of the superposition and
binding. A partial implementation is possible, but will prevent the usage of
certain parts of Nengo SPA. For example, when not providing neural
implementations, only non-neural math can be performed.
