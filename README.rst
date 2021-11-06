.. image:: https://img.shields.io/pypi/v/nengo-spa.svg
  :target: https://pypi.org/project/nengo-spa
  :alt: Latest PyPI version

.. image:: https://img.shields.io/pypi/pyversions/nengo-spa.svg
  :target: https://pypi.org/project/nengo-spa
  :alt: Python versions

.. image:: https://img.shields.io/codecov/c/github/nengo/nengo-spa/master.svg
  :target: https://codecov.io/gh/nengo/nengo-spa/branch/master
  :alt: Test coverage

|

.. image:: https://www.nengo.ai/design/_images/nengo-spa-full-light.svg
   :alt: NengoSPA logo
   :width: 300px

*************************************************************
Implementation of the Semantic Pointer Architecture for Nengo
*************************************************************

The `Semantic Pointer Architecture
<https://www.nengo.ai/nengo-spa/user_guide/spa_intro.html>`_ provides an
approach to building cognitive models implemented with large-scale spiking
neural networks.

Feature highlights
==================

- Write arbitrarily complex expressions with type checking involving neurally
  represented and static Semantic Pointers like
  ``dot((role * filler + BiasVector) * tag, cmp) >> target``. NengoSPA will
  take care of implementing the required neural networks.
- Quickly implement action selection systems based on a biological plausible
  model of the basal ganglia and thalamus.
- Neural representations are optimized for representing Semantic Pointers.
- Support for using different binding methods with algebras. NengoSPA ships
  with implementations of circular convolution (default) and vector-derived
  transformation binding (VTB), which is particularly suitable for deep
  structures. Different binding operations/algebras can be mixed in a single
  model.
- Seamless integration with non-SPA Nengo models.
- Binding powers and fractional bindings.


Project status
==============

- All of the core functionality is implemented and the API is stable.
- While basic integration with the NengoGUI works, it should be improved in
  the future. However, this will not be pursued until major improvements to
  NengoGUI are released.


Installation
============

NengoSPA depends on `Nengo 2.7+ <https://nengo.github.io/>`_, and we recommend
that you install Nengo before installing NengoSPA.

To install NengoSPA::

    pip install nengo-spa

NengoSPA is tested to work on Python 3.6+.

If you need support for Python 2.7 or 3.4, NengoSPA 0.6.2 is the last version
with support for these earlier Python versions.


Documentation
=============

The documentation can be found `here <https://www.nengo.ai/nengo-spa/>`_.

Getting Help
============

Questions relating to Nengo and NengoSPA, whether it's use or it's
development, should be asked on the Nengo forum at `<https://forum.nengo.ai>`_.
