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
<https://www.nengo.ai/nengo/index.html>`_ (which depends on `NumPy
<http://www.numpy.org/>`_). Please check the `Nengo documentation
<https://www.nengo.ai/nengo/getting_started.html#installation>`__ for
instructions on how to install Nengo or NumPy.

We also suggest that you install `SciPy <https://www.scipy.org/>`_ to obtain the
best accuracy in your networks and access to all Nengo SPA features. We
recommend installing SciPy with the same method that you used to install NumPy.
Check the `Nengo documentation
<https://www.nengo.ai/nengo/getting_started.html#installing-numpy>`__ and
`SciPy documentation <https://www.scipy.org/install.html>`_ for available
installation methods.

Further optional packages
^^^^^^^^^^^^^^^^^^^^^^^^^

Besides optional SciPy package, there are a few more optional packages that
are only required for specific purposes.

.. code-block:: bash

   pip install nengo[docs]  # For building docs
   pip install nengo[tests]  # For running the test suite
   pip install nengo[all]  # All of the above, including SciPy

Examples
^^^^^^^^

Examples of Nengo SPA usage are included in this documentation. Each example
has a link to download it as Jupyter notebook at the top of its page.


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

* If you are unfamiliar with the Semantic Pointer Architecture, start with the
  :ref:`/user_guide/spa_intro.rst`.
* If you are new to building SPA models with Nengo, read the
  :ref:`/examples/intro.ipynb`.
* If you have used the legacy SPA module included in core Nengo, read the
  tutorial on :ref:`/examples/intro_coming_from_legacy_spa.ipynb`.
