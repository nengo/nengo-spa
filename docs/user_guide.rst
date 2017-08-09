User Guide
==========


Introduction to the Semantic Pointer Architecture
-------------------------------------------------

Briefly, the Semantic Pointer hypothesis states:

    Higher-level cognitive functions in biological systems are made possible by
    Semantic Pointers. Semantic Pointers are neural representations that carry
    partial semantic content and are composable into the representational
    structures necessary to support complex cognition.

The term ‘Semantic Pointer’ was chosen because the representations in the
architecture are like ‘pointers’ in computer science (insofar as they can be
‘dereferenced’ to access large amounts of information which they do not
directly carry). However, they are ‘semantic’ (unlike pointers in computer
science) because these representations capture relations in a semantic vector
space in virtue of their distances to one another, as typically envisaged by
connectionists.

The book `How to build a brain
<https://www.amazon.com/How-Build-Brain-Architecture-Architectures/dp/0199794545>`_
from Oxford University Press gives a broader introduction into the Semantic
Pointer Architecture (SPA) and its use in cognitive modeling. To describe the
architecture, the book covers four main topics that are semantics, syntax,
control, and learning and memory. The discussion of semantics considers how
Semantic Pointers are generated from information that impinges on the senses,
reproducing details of the spiking tuning curves in the visual system. Syntax is
captured by demonstrating how very large structures can be encoded, by binding
Semantic Pointers to one another. The section on control considers the role of
the basal ganglia and other structures in routing information throughout the
brain to control cognitive tasks. The section on learning and memory describes
how the SPA includes adaptability (despite this not being a focus of the `Neural
Engineering Framework <http://compneuro.uwaterloo.ca/research/nef.html>`_ used
in Nengo) showing how detailed spiking timing during learning can be
incorporated into the basal ganglia model using a biologically plausible
STDP-like rule.


Structured representations
^^^^^^^^^^^^^^^^^^^^^^^^^^




Introduction to Nengo SPA
-------------------------
