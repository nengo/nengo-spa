Getting started
===============

Installation
------------

To install Nengo SPA, we recommend using ``pip``::

    pip install nengo_spa

``pip`` will do its best to install all of Nengo's requirements when it
installs Nengo SPA. However, if anything goes wrong during this process, you
can install Nengo SPA's requirements manually before doing ``pip install
nengo_spa`` again.

Nengo SPA only depends on `core Nengo
<https://pythonhosted.org/nengo/index.html>`_ (which depends on `NumPy
<http://www.numpy.org/>`_). Please check the `Nengo documenation
<https://pythonhosted.org/nengo/getting_started.html#installation>`_ for
instructions on how to install Nengo or NumPy.

We also suggest that you install `SciPy <https://www.scipy.org/>`_ to obtain the
best accuracy in your networks and access to all Nengo SPA features. We
recommend installing SciPy with the same method that you used to install NumPy.
Check the `Nengo documentation
<https://pythonhosted.org/nengo/getting_started.html#installing-numpy>`_ and
`SciPy documentation <https://www.scipy.org/install.html>`_ for available
installation methods.

Further optional packages
^^^^^^^^^^^^^^^^^^^^^^^^^

Besides optional SciPy package, there are a few more optional packages that
are only required for specific purposes.

* To run the test suite the packages listed in ``requirements-test.txt`` need to
  be installed.
* To build the documentation the packages listed in ``requirements-docs.txt``
  need to be installed


Usage
-----

The recommended way to use Nengo SPA is to ``import nengo_spa as spa``. (Note
that this uses an underscore in the module name and is different from
`nengo.spa` which refers to the legacy SPA module shipped with core Nengo.)

To use Nengo SPA functionality in your models use `nengo_spa.Network` instead of
`nengo.Network`. A basic Nengo SPA model can look like this::

    import nengo
    import nengo_spa as spa

    with spa.Network() as model:
        state = spa.State(16)
        spa.sym.A >> state


Next steps
----------

* If you are unfamiliar with the Semantic Pointer Architecture, TODO
* If you are new to building SPA models with Nengo, TODO
* If you have used the legacy SPA module included in core Nengo, TODO
