# -*- coding: utf-8 -*-

import enum

from pydantic import confloat, conint

from ._base import MPE_MODULE, ImmutableDataObject
from ._helpers import set_module


@set_module(MPE_MODULE)
class FastMarchingMethodOrder(enum.IntEnum):
    first = 1
    second = 2


@set_module(MPE_MODULE)
class ExtractPointUpdateMethod(str, enum.Enum):
    euler = 'euler'
    runge_kutta = 'runge_kutta'


@set_module(MPE_MODULE)
class Parameters(ImmutableDataObject):
    """MPE algorithm parameters
    """

    fmm_grid_spacing: confloat(strict=True, gt=0.0) = 1.0
    fmm_order: FastMarchingMethodOrder = FastMarchingMethodOrder.first
    extract_grid_spacing: confloat(strict=True, gt=0.0) = 1.0
    extract_max_iterations: conint(strict=True, ge=100) = 2000
    extract_point_update_method: ExtractPointUpdateMethod = ExtractPointUpdateMethod.runge_kutta
    travel_time_cache: bool = False
