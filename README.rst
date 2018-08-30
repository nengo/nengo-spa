.. image:: https://img.shields.io/travis/nengo/nengo-spa/master.svg
  :target: https://travis-ci.org/nengo/nengo-spa
  :alt: Travis-CI build status

.. image:: https://ci.appveyor.com/api/projects/status/5blofvbh8305w75g/branch/master?svg=true
  :target: https://ci.appveyor.com/project/jgosmann/nengo-spa/branch/master
  :alt: AppVeyor build status

.. image:: https://img.shields.io/codecov/c/github/nengo/nengo-spa/master.svg
  :target: https://codecov.io/gh/nengo/nengo-spa/branch/master
  :alt: Test coverage

|

.. image:: https://www.nengo.ai/design/_images/full-light2.svg
   :alt: Nengo SPA logo
   :width: 300px

*************************************************************
Implementation of the Semantic Pointer Architecture for Nengo
*************************************************************

The `Semantic Pointer Architecture
<https://www.nengo.ai/nengo-spa/user_guide/spa_intro.html>`_ provides an
approach to building cognitive models implemented with large-scale spiking
neural networks.

Project status
==============

- All of the core functionality is implemented and most of the API should be
  fairly stable.
- While basic integration with the Nengo GUI works, it should be improved in
  the future. However, those improvements will mostly depend on Nengo GUI
  providing an appropriate plugin system.

Installation
============

Nengo SPA depends on `Nengo 2.7+ <https://nengo.github.io/>`_, and we recommend
that you install Nengo before installing Nengo SPA.

To install Nengo SPA::

    pip install nengo-spa

Nengo SPA is tested to work on Python 2.7 and 3.4+.

Documentation
=============

The documentation can be found `here <https://www.nengo.ai/nengo-spa/>`_.

Getting Help
============

Questions relating to Nengo and Nengo SPA, whether it's use or it's
development, should be asked on the Nengo forum at `<https://forum.nengo.ai>`_.
