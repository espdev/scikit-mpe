.. _api:

*************
API Reference
*************

.. currentmodule:: skmpe

API Summary
===========

.. autosummary::
    :nosignatures:

    InitialInfo
    PathInfo
    PathInfoResult

    TravelTimeOrder
    OdeSolverMethod
    Parameters
    parameters
    default_parameters

    MPEError
    ComputeTravelTimeError
    PathExtractionError
    EndPointNotReachedError

    PathExtractionResult
    MinimalPathExtractor
    mpe

|

Data and Models
===============

.. autoclass:: InitialInfo
    :members:

.. autoclass:: PathInfo
    :show-inheritance:

.. autoclass:: PathInfoResult
    :show-inheritance:

Parameters
==========

.. autoclass:: TravelTimeOrder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: OdeSolverMethod
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: Parameters
    :members:
    :undoc-members:

.. autofunction:: parameters
.. autofunction:: default_parameters

Exceptions
==========

.. autoclass:: MPEError
    :show-inheritance:

.. autoclass:: ComputeTravelTimeError
    :show-inheritance:

.. autoclass:: PathExtractionError
    :members:
    :show-inheritance:

.. autoclass:: EndPointNotReachedError
    :members:
    :inherited-members: PathExtractionError
    :exclude-members: with_traceback
    :show-inheritance:

Path Extraction
===============

.. autoclass:: PathExtractionResult
    :show-inheritance:

.. autoclass:: MinimalPathExtractor
    :members:
    :special-members: __call__
