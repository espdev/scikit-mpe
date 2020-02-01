# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np
import skfmm as fmm

from ._base import mpe as mpe_impl_dispatch
from ._base import PointType, WayPointsType, InitialInfo, Parameters, ResultPathInfo


@mpe_impl_dispatch.register(np.ndarray)
def mpe(speed_data: np.ndarray, *,
        start_point: PointType,
        end_point: PointType,
        way_points: Optional[WayPointsType] = None,
        params: Optional[Parameters] = None) -> ResultPathInfo:
    init_info = InitialInfo(
        speed_data=speed_data,
        start_point=start_point,
        end_point=end_point,
        way_points=way_points
    )

    return mpe(init_info, params=params)


@mpe_impl_dispatch.register(InitialInfo)
def mpe(init_info: InitialInfo, *,
        params: Optional[Parameters] = None) -> ResultPathInfo:
    if params is None:
        params = Parameters()

    raise NotImplementedError
