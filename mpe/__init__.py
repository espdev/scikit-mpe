# -*- coding: utf-8 -*-

__version__ = '0.1.0'

from ._base import InitialInfo, Parameters, mpe

# register dispatchered implementation
import mpe._impl as _impl  # noqa


__all__ = [
    'InitialInfo',
    'Parameters',
    'mpe',
]
