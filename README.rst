.. image:: https://img.shields.io/travis/nengo/nengo-spa/master.svg
  :target: https://travis-ci.org/nengo/nengo-spa
  :alt: Travis-CI build status

.. image:: https://img.shields.io/codecov/c/github/nengo/nengo-spa/master.svg
  :target: https://codecov.io/gh/nengo/nengo-spa/branch/master
  :alt: Test coverage

|

.. image:: https://www.nengo.ai/design/_images/nengo-spa-full-light.svg
   :alt: Nengo SPA logo
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
  ``dot((role * filler + BiasVector) * tag, cmp) >> target``. Nengo SPA will
  take care of implementing the required neural networks.
- Quickly implement action selection systems based on a biological plausible
  model of the basal ganglia and thalamus.
- Neural representations are optimized for representing Semantic Pointers.
- Support for using different binding methods with algebras. Nengo SPA ships
  with implementations of circular convolution (default) and vector-derived
  transformation binding (VTB), which is particularly suitable for deep
  structures. Different binding operations/algebras can be mixed in a single
  model.
- Seamless integration with non-SPA Nengo models.


Project status
==============

- All of the core functionality is implemented and the API is stable.
- While basic integration with the Nengo GUI works, it should be improved in
  the future. However, this will not be pursued until major improvements to
  Nengo GUI are released.


Installation
============

Nengo SPA depends on `Nengo 2.7+ <https://nengo.github.io/>`_, and we recommend
that you install Nengo before installing Nengo SPA.

To install Nengo SPA::

    pip install nengo-spa

Nengo SPA is tested to work on Python 3.5+.

If you need support for Python 2.7 or 3.4, Nengo SPA 0.6.2 is the last version
with support for these earlier Python versions.


Documentation
=============

The documentation can be found `here <https://www.nengo.ai/nengo-spa/>`_.

Getting Help
============

Questions relating to Nengo and Nengo SPA, whether it's use or it's
development, should be asked on the Nengo forum at `<https://forum.nengo.ai>`_.
