# -*- coding: utf-8 -*-

from ._base import (
    InitialInfo,
    FastMarchingMethodOrder,
    ExtractPointUpdateMethod,
    Parameters,
    PathInfo,
    ResultPathInfo,
    mpe
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
    'FastMarchingMethodOrder',
    'ExtractPointUpdateMethod',
    'Parameters',
    'PathInfo',
    'ResultPathInfo',

    'MPEError',
    'ComputeTravelTimeError',
    'ExtractPathError',
    'EndPointNotReachedError',

    'mpe',
]
