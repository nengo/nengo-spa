Contributing to Nengo SPA
=========================

Issues and pull requests are always welcome! We appreciate help from the
community to make Nengo SPA better.

Filing issues
-------------

If you find a bug in Nengo SPA, or think that a certain feature is missing,
please consider `filing an issue <https://github.com/nengo/nengo-spa/issues>`_.
Please search the currently open issues first to see if your bug or feature
request already exists. If so, feel free to add a comment to the issue
so that we know that multiple people are affected.

Making pull requests
--------------------

If you want to fix a bug or add a feature to Nengo SPA, we welcome pull
requests.  We try to maintain 100% test coverage, so any new features should
also include unit tests to cover that change.  If you fix a bug it's also a good
idea to add a unit test, so the bug doesn't get un-fixed in the future!


Building the documentation
--------------------------

To build the documentation install the required dependencies by running
the following command from the root folder of the Nengo SPA source code:

.. code-block:: bash

   python setup.py -e .[docs]

To build the documentation use one of the following commands:

.. code-block:: bash

   python setup.py build_sphinx

or if you need set explicitly set the Jupyter kernel for building the
notebooks included in the documentation:

.. code-block:: bash

   sphinx-build docs docs/_build -D nbsphinx_kernel_name=<kernelname>

You will find the build documentation in the ``docs/_build`` folder.


Contributor agreement
---------------------

We require that all contributions be covered under our contributor assignment
agreement. Please see `the agreement <https://www.nengo.ai/caa.html>`_
for instructions on how to sign.
