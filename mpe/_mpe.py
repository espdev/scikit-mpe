# -*- coding: utf-8 -*-

from typing import Optional, Tuple

from pydantic import BaseModel, Extra, validator, root_validator, confloat, conint

import numpy as np
import skfmm as fmm


PointType = Tuple[int, ...]

MIN_NDIM = 2


class ImmutableDataObject(BaseModel):
    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_mutation = False


class InitialInfo(ImmutableDataObject):
    """
    """

    speed_data: np.ndarray

    start_point: PointType
    end_point: PointType
    way_points: Optional[Tuple[PointType, ...]] = None

    @root_validator
    def _check_ndim(cls, values):
        speed_data = values.get('speed_data')

        start_point = values.get('start_point')
        end_point = values.get('end_point')
        way_points = values.get('way_points')

        if speed_data is None or start_point is None or end_point is None:
            return values

        ndim = speed_data.ndim

        if ndim < MIN_NDIM:
            raise ValueError(f'minimum dimension must be {MIN_NDIM}.')

        if ndim != len(start_point) or ndim != len(end_point):
            raise ValueError(f"the start and end points must have dimension {ndim}.")

        if way_points is not None:
            for point in way_points:
                if len(point) != ndim:
                    raise ValueError(f"the start, end and all way points must have dimension {ndim}.")

        return values

    @validator('start_point', 'end_point', 'way_points')
    def _check_points(cls, v, field, values):
        if v is None:
            return v

        if 'speed_data' not in values:
            return v

        speed_data = values['speed_data']

        def validate(point, index=None):
            if len(point) != speed_data.ndim:
                # this is the concern of another validator
                return point

            n_point = f' [{index}] ' if index else ' '

            for i, (n_coord, n_size) in enumerate(zip(point, speed_data.shape)):
                if n_coord < 0 or n_coord >= n_size:
                    raise ValueError(f"'{field.name}' {point}{n_point}coordinate {i} "
                                     f"is out of 'speed_data' bounds [0, {n_size}).")

            if speed_data[point] is np.ma.masked:
                raise ValueError(f"'{field.name}' {point}{n_point}inside 'speed_data' masked area.")

        if field.name == 'way_points':
            for idx, p in enumerate(v):
                validate(p, idx)
        else:
            validate(v)

        return v


class Parameters(ImmutableDataObject):
    fmm_grid_spacing: confloat(gt=0.0) = 1.0
    extract_grid_spacing: confloat(gt=0.0) = 1.0
    max_iterations: conint(ge=100) = 1000


class ResultPathInfo(ImmutableDataObject):
    full_path: np.ndarray
    travel_times: Tuple[np.ndarray, ...]


def mpe(init_info: InitialInfo, params: Optional[Parameters] = None) -> ResultPathInfo:
    """

    Parameters
    ----------
    init_info
    params

    Returns
    -------

    """

    if params is None:
        params = Parameters()
