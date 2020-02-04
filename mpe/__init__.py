# -*- coding: utf-8 -*-

from ._base import (
    InitialInfo,
    PathInfo,
    ResultPathInfo,
    mpe
)

from ._parameters import (
    TravelTimeComputeOrder,
    Parameters,
    parameters,
    default_parameters,
)

from ._exceptions import (
    MPEError,
    ComputeTravelTimeError,
    PathExtractionError,
    EndPointNotReachedError,
)

from ._mpe import MinimalPathExtractorBase, RungeKuttaMinimalPathExtractor

# register dispatchered API
import mpe._api as _api  # noqa


__version__ = '0.1.0'

__all__ = [
    'InitialInfo',
    'PathInfo',
    'ResultPathInfo',

    'TravelTimeComputeOrder',
    'Parameters',
    'parameters',
    'default_parameters',

    'MPEError',
    'ComputeTravelTimeError',
    'PathExtractionError',
    'EndPointNotReachedError',

    'MinimalPathExtractorBase',
    'RungeKuttaMinimalPathExtractor',

    'mpe',
]
