"""SPA modules.

SPA modules derive from `nengo_spa.Network` and thus provide information about
inputs and outputs that can be used in action rules and the associated
vocabularies. Note that a module might have no inputs and outputs that can be
used in action rules, but only inputs and outputs that must be manually
connected to. Many SPA modules are networks that might be automatically
created by building action rules. Because of this, it is possible to set module
parameters with `nengo.Config` objects to allow to easily change parameters
of networks created in this way.

Note that SPA modules can be used as standalone networks without using
nengo_spa features.
"""
