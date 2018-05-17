***************
Release History
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed


0.4.0 (May 17, 2018)
====================

This release increases the minimum required Nengo version to Nengo 2.7
(previously Nengo 2.4).

**Added**

- Added documentation and build tools for the documentation.
  (`#68 <https://github.com/nengo/nengo_spa/pull/68>`_)

**Changed**

- This release introduces a new syntax for SPA action rules.
  (`#114 <https://github.com/nengo/nengo_spa/pull/114>`_)

**Remove**

- Unnecessary ``vocab`` argument from ``Transcode``.
  (`#68 <https://github.com/nengo/nengo_spa/pull/68>`_)

**Fixed**

- Validation of ``VocabOrDimParam`` and ``VocabularyMapParam``.
  (`#95 <https://github.com/nengo/nengo_spa/issues/95>`_,
  `#98 <https://github.com/nengo/nengo_spa/pull/98>`_)
- Allow the configuration of instance parameters with
  ``nengo_spa.Network.config``.
  (`#112 <https://github.com/nengo/nengo_spa/issues/112>`_,
  `#113 <https://github.com/nengo/nengo_spa/pull/113>`_)
- Fix an undeclared input to the ``IAAssocMem`` module.
  (`#118 <https://github.com/nengo/nengo_spa/issues/118>`_,
  `#120 <https://github.com/nengo/nengo_spa/pull/120>`_)


0.3.2 (November 17, 2017)
=========================

**Added**

- Add ``all_bgs`` and ``all_thals`` methods to
  ``AstAccessor`` to enable easy access to these objects.
  (`#61 <https://github.com/nengo/nengo_spa/pull/99>`__,
  `#28 <https://github.com/nengo/nengo_spa/issues/80>`__)

**Fixed**

- Allow the ``spa.Actions`` string to be empty.
  (`#107 <https://github.com/nengo/nengo_spa/issues/107>`_,
  `#109 <https://github.com/nengo/nengo_spa/pull/109>`_)
- The ``pass`` keyword can now be used to create blocks in action rules that
  do not have any effect.
  (`#101 <https://github.com/nengo/nengo_spa/issues/101>`_,
  `#103 <https://github.com/nengo/nengo_spa/pull/103>`_)
- Allow comments at various places in actions rules.
  (`#102 <https://github.com/nengo/nengo_spa/issues/102>`_,
  `#104 <https://github.com/nengo/nengo_spa/pull/104>`_)


0.3.1 (November 7, 2017)
========================

**Changed**

- Clearer error message as a ``SpaTypeError`` something is used as input/output
  in an action rule without being declared as such.
  (`#82 <https://github.com/nengo/nengo_spa/issues/82>`_,
  `#89 <https://github.com/nengo/nengo_spa/pull/89>`_)

**Fixed**

- Allow leading comments in actions rules.
  (`#81 <https://github.com/nengo/nengo_spa/issues/81>`_,
  `#85 <https://github.com/nengo/nengo_spa/pull/85>`_)
- Gave the basal ganglia a default label.
  (`#84 <https://github.com/nengo/nengo_spa/issues/84>`_,
  `#88 <https://github.com/nengo/nengo_spa/pull/88>`_)
- Fixed warning produce by the ``create_inhibit_node`` function.
  (`#90 <https://github.com/nengo/nengo_spa/pull/90>`_)
- Prevent whitespace from being completely removed in action rules.
  (`#92 <https://github.com/nengo/nengo_spa/issues/92>`_,
  `#93 <https://github.com/nengo/nengo_spa/pull/93>`_)
- Have the ``intercept_width`` argument of ``IA`` actually take effect.
  (`#94 <https://github.com/nengo/nengo_spa/issues/94>`_,
  `#97 <https://github.com/nengo/nengo_spa/pull/97>`_)


0.3.0 (October 16, 2017)
========================

**Added**

- Add ``add_output`` and ``add_neuron_output`` methods to
  ``IdentityEnsembleArray`` to provide the full API that is provided by the
  regular Nengo ``EnsembleArray``.
  (`#61 <https://github.com/nengo/nengo_spa/pull/61>`_,
  `#28 <https://github.com/nengo/nengo_spa/issues/28>`_)
- Add ``create_inhibit_node`` function to create nodes that inhibit complete
  Nengo networks.
  (`#65 <https://github.com/nengo/nengo_spa/pull/65>`_,
  `#26 <https://github.com/nengo/nengo_spa/issues/26>`_)
- Add a ``solver`` argument to the action rule's ``translate`` to use a solver
  instead of an outer product to obtain the transformation matrix which can
  give slightly better results.
  (`#62 <https://github.com/nengo/nengo_spa/pull/62>`_,
  `#57 <https://github.com/nengo/nengo_spa/issues/57>`_)

**Changed**

- Actions rules do not require module to be assigned to the model any longer.
  They will access exactly the same variables as are available in the
  surrounding Python code. This means that existing action rules need to be
  changed to reference the correct names.
  (`#63 <https://github.com/nengo/nengo_spa/pull/63>`_)
- The action rule syntax changed significantly.
  (`#54 <https://github.com/nengo_spa/nengo/issues/54>`_,
  `#72 <https://github.com/nengo_spa/nengo/pull/72>`_)
- Actions will be build automatically without an explicit call to ``build()``.
  (`#59 <https://github.com/nengo/nengo_spa/pull/59>`_,
  `#45 <https://github.com/nengo/nengo_spa/issues/45>`_,
  `#55 <https://github.com/nengo/nengo_spa/issues/55>`_)
- Consolidated the functionality of ``Encode`` and ``Decode`` into
  ``Transcode``.
  (`#67 <https://github.com/nengo/nengo_spa/pull/67>`_,
  `#58 <https://github.com/nengo/nengo_spa/issues/58>`_)

**Fixed**

- Fix some operations changing the dimensionality of semantic pointers with an
  odd initial dimensionality.
  (`#52 <https://github.com/nengo/nengo_spa/issues/52>`_,
  `#53 <https://github.com/nengo/nengo_spa/pull/53>`_)
- When building actions the basal ganglia and thalamus will only be created
  when actually required.
  (`#60 <https://github.com/nengo/nengo_spa/pull/60>`_,
  `#42 <https://github.com/nengo/nengo_spa/issues/42>`_)
- The vocabulary translate mechanism will properly ignore missing keys in the
  target vocabulary when ``populate=False``.
  (`#62 <https://github.com/nengo/nengo_spa/pull/62>`_,
  `#56 <https://github.com/nengo/nengo_spa/issues/56>`_)
- Allow empty string as argument to `Vocabulary.populate`.
  (`#73 <https://github.com/nengo_spa/nengo/pull/73>`_)


0.2 (June 22, 2017)
===================

**Added**

- Tutorial explaining what has changed in nengo_spa compared to the legacy SPA
  implementation.
  (`#46 <https://github.com/nengo/nengo_spa/pull/46>`_)
- Examples can be extracted with ``python -m nengo_spa extract-examples
  <destination>``.
  (`#49 <https://github.com/nengo/nengo_spa/pull/49>`_,
  `#7 <https://github.com/nengo/nengo_spa/issues/7>`_)

**Changed**

- Replaced *input_keys* and *output_keys* arguments of associative memories
  with a single *mapping* argument.
  (`#29 <https://github.com/nengo/nengo_spa/pull/29>`_,
  `#8 <https://github.com/nengo/nengo_spa/issues/8>`_)
- Replaced *ampa_config* and *gaba_config* parameters of the
  *BasalGanglia* with *ampa_synapse* and *gaba_synapse* parameters.
  Removed the *general_config* parameter.
  (`#30 <https://github.com/nengo/nengo_spa/pull/30>`_,
  `#23 <https://github.com/nengo/nengo_spa/issues/23>`_)

**Fixed**

- Improved a number of error messages.
  (`#35 <https://github.com/nengo/nengo_spa/pull/35>`_,
  `#32 <https://github.com/nengo/nengo_spa/issues/32>`_,
  `#34 <https://github.com/nengo/nengo_spa/issues/34>`_)
- Improved accuracy by fixing choice of evaluation point and intercept
  distributions.
  (`#39 <https://github.com/nengo/nengo_spa/pull/39>`_)
- Correctly apply transforms on first vector in vocabularies on on non-strict
  vocabularies.
  (`#43 <https://github.com/nengo/nengo_spa/pull/43>`_)


0.1.1 (May 19, 2017)
====================

**Fixed**

- Updated the 0.1 changelog.


0.1 (May 19, 2017)
==================

Initial release of Nengo SPA with core functionality, but excluding

- updates and completion the documentation,
- proper integration with Nengo GUI.

The API is still conisdered unstable in some parts of it are likely to change
in the future.

Main features compared to the SPA implementation shipped with Nengo are:

- neural representations have been optimized for higher accuracy,
- support for arbitrarily complex action rules,
- SPA networks can be used as normal Nengo networks,
- and SPA networks can be nested.
