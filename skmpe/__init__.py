# -*- coding: utf-8 -*-

from ._base import (
    InitialInfo,
    PathInfo,
    PathInfoResult,
    mpe
)

from ._parameters import (
    TravelTimeOrder,
    OdeSolverMethod,
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

from ._mpe import (
    PathExtractionResult,
    MinimalPathExtractor,
)

# register dispatchered API
import skmpe._api as _api  # noqa


__version__ = '0.1.1'

__all__ = [
    'InitialInfo',
    'PathInfo',
    'PathInfoResult',

    'TravelTimeOrder',
    'OdeSolverMethod',
    'Parameters',
    'parameters',
    'default_parameters',

    'MPEError',
    'ComputeTravelTimeError',
    'PathExtractionError',
    'EndPointNotReachedError',

    'PathExtractionResult',
    'MinimalPathExtractor',
    'mpe',
]
