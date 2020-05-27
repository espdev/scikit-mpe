# -*- coding: utf-8 -*-

from importlib_metadata import metadata, PackageNotFoundError

from ._base import (
    InitialInfo,
    PathInfo,
    PathInfoResult,
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

from skmpe._api import mpe


try:
    __version__ = metadata('scikit-mpe')['version']
except PackageNotFoundError:  # pragma: no cover
    __version__ = '0.0.0.dev'


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
