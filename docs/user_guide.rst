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

The Semantic Pointer Architecture (SPA) uses a specific type of a Vector
Symbolic Architecture. That means it uses (high-dimensional) vectors to
represent concepts. These can be combined with certain linear and non-linear
operations to bind the concept vectors and build structured representations.

The specific operations used by the SPA where first suggested by Tony A. Plate
in `Holographic Reduced Representation: Distributed Representation for Cognitive
Structures
<https://www.amazon.ca/Holographic-Reduced-Representation-Distributed-Structures-ebook/dp/B0188Y14VS/ref=sr_1_1?ie=UTF8&qid=1502311400&sr=8-1>`_
In particular, we usually use random vectors of unit-length and three basic
operations.

1. *Superposition*: Two vectors :math:`\vec{v}` and :math:`\vec{w}` can be
   combined in a union-like operation by simple addition as :math:`\vec{u}
   = \vec{v} + \vec{w}`. The resulting vector will be similar to both of the
   original vectors.
2. *Binding*: The binding has to produce a vector that is dissimilar to both of
   the original vectors and allows to recover one of the original vectors given
   the other one. In the SPA, we employ circular convolution for this purpose
   defined as

   .. math::
      \vec{u} = \vec{v} \circledast \vec{w}\ :\quad u_i = \sum_{j=1}^D v_j
      w_{(i-j)\ \mathrm{mod}\ D}

   where :math:`D` is the dimensionality of the vectors.
3. *Unbinding*: To unbind a vector from a circular convolution, another circular
   convolution with the approximate inverse of one of the vectors can be used:
   :math:`\vec{v} \approx \vec{u} \circledast \vec{w}^+`. The approximate
   inverse is given by reordering the vector components: :math:`\vec{w}^+
   = (w_1, w_D, w_{D-1}, \dots, w_2)^T`.

Note that circular convolution is associative, commutative, and distributive:

* :math:`(\vec{u} \circledast \vec{v}) \circledast \vec{w} = \vec{u} \circledast
  (\vec{v} \circledast \vec{w})`,
* :math:`\vec{v} \circledast \vec{w} = \vec{w} \circledast \vec{v}`,
* :math:`\vec{u} \circledast (\vec{v} + \vec{w}) = \vec{u} \circledast \vec{v}
  + \vec{u} \circledast \vec{w}`.

Let us consider a simple example: Given vectors for *red*, *blue*,
*square*, and *circle*, we can represent a scene with a *red square* and *blue
circle* as

.. math::
   \vec{v} = \mathrm{Red} \circledast \mathrm{Square} + \mathrm{Blue}
   \circledast \mathrm{Circle}

If we want to know the color of the square, we can unbind the *square* vector:

.. math:: \vec{v} \circledast \mathrm{Square}^+ = \mathrm{Red} \circledast
   \mathrm{Square} \circledast \mathrm{Square}^+ + \mathrm{Blue} \circledast
   \mathrm{Circle} \circledast \mathrm{Square}^+ \approx \mathrm{Red}
   + \mathit{noise}

TODO: link to more in depth discussion of circular convolution (including
unitary vectors etc); mention alternate binding approaches?


Spaun
^^^^^

TODO


Introduction to Nengo SPA
-------------------------


Writing modules
---------------
