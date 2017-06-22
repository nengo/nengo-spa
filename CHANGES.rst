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
