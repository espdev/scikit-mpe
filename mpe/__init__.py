# -*- coding: utf-8 -*-

from ._base import (
    InitialInfo,
    PathInfo,
    ResultPathInfo,
    mpe
)

from ._parameters import (
    FastMarchingMethodOrder,
    PathExtractionMethod,
    Parameters,
    parameters,
    default_parameters,
)

from ._exceptions import (
    MPEError,
    ComputeTravelTimeError,
    ExtractPathError,
    EndPointNotReachedError,
)

# register dispatchered implementation
import mpe._impl as _impl  # noqa


__version__ = '0.1.0'

__all__ = [
    'InitialInfo',
    'PathInfo',
    'ResultPathInfo',

    'FastMarchingMethodOrder',
    'PathExtractionMethod',
    'Parameters',
    'parameters',
    'default_parameters',

    'MPEError',
    'ComputeTravelTimeError',
    'ExtractPathError',
    'EndPointNotReachedError',

    'mpe',
]
