nengo\_spa
==========

Commonly used classes and functions are accessible at the top level of the
`nengo_spa` package.


.. autosummary::

   nengo_spa.actions.ActionSelection
   nengo_spa.network.create_inhibit_node
   nengo_spa.network.ifmax
   nengo_spa.network.Network
   nengo_spa.vocab.Vocabulary


.. data:: nengo_spa.sym

   Provides Semantic Pointer symbols for symbolic expressions that are not tied
   to a single vocabulary. The vocabulary will be determined from the context.
   To use a symbol access it as an attribute on this object.

   For example the following::

       sym.A * sym.B >> state

   is equivalent to::

       state.vocab.parse('A * B') >> state


Modules
-------

.. autosummary::

   nengo_spa.modules.AssociativeMemory
   nengo_spa.modules.IAAssocMem
   nengo_spa.modules.ThresholdingAssocMem
   nengo_spa.modules.WTAAssocMem
   nengo_spa.modules.BasalGanglia
   nengo_spa.modules.Bind
   nengo_spa.modules.Compare
   nengo_spa.modules.Product
   nengo_spa.modules.Scalar
   nengo_spa.modules.State
   nengo_spa.modules.Thalamus
   nengo_spa.modules.Transcode


Examination of Semantic Pointers
--------------------------------

.. autosummary::

   nengo_spa.examine.pairs
   nengo_spa.examine.similarity
   nengo_spa.examine.text


Operators
---------

.. autosummary::

   nengo_spa.operators.dot
   nengo_spa.operators.reinterpret
   nengo_spa.operators.translate
