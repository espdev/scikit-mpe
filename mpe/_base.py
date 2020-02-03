# -*- coding: utf-8 -*-

import collections
import logging
from typing import Optional, Tuple, Sequence, List, NamedTuple, overload, TYPE_CHECKING

from pydantic import BaseModel, Extra, validator, root_validator
import numpy as np

from ._helpers import set_module, singledispatch

if TYPE_CHECKING:
    from ._parameters import Parameters


PointType = Sequence[int]
FloatPointType = Sequence[float]
WayPointsType = Sequence[PointType]
PointTypeModel = Tuple[int, ...]
WayPointsTypeModel = Tuple[PointTypeModel, ...]


MPE_MODULE = 'mpe'
MIN_NDIM = 2


logger = logging.getLogger(MPE_MODULE)
logger.addHandler(logging.NullHandler())


class ImmutableDataObject(BaseModel):
    """Base immutable data object with validating fields
    """

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_mutation = False


@set_module(MPE_MODULE)
class InitialInfo(ImmutableDataObject):
    """Initial info for extracting path
    """

    speed_data: np.ndarray

    start_point: PointTypeModel
    end_point: PointTypeModel
    way_points: WayPointsTypeModel = ()

    @root_validator
    def _check_ndim(cls, values):
        speed_data = values.get('speed_data')

        start_point = values.get('start_point')
        end_point = values.get('end_point')
        way_points = values.get('way_points')

        if (speed_data is None or
                start_point is None or
                end_point is None or
                way_points is None):
            return values

        ndim = speed_data.ndim

        if ndim < MIN_NDIM:
            raise ValueError(f'minimum dimension must be {MIN_NDIM}.')

        if ndim != len(start_point) or ndim != len(end_point):
            raise ValueError(f"the start and end points must have dimension {ndim}.")

        for point in way_points:
            if len(point) != ndim:
                raise ValueError(f"the start, end and all way points must have dimension {ndim}.")

        return values

    @root_validator
    def _check_point_duplicates(cls, values):
        start_point = values.get('start_point')
        end_point = values.get('end_point')
        way_points = values.get('way_points')

        if start_point is None or end_point is None or way_points is None:
            return values

        all_points = [start_point, end_point, *way_points]
        duplicates = [point for point, count in collections.Counter(all_points).items() if count > 1]

        if duplicates:
            raise ValueError(
                f'the points must not be duplicated, there are duplicated points: {duplicates}')

        return values

    @validator('start_point', 'end_point', 'way_points')
    def _check_points(cls, v, field, values):
        if v is None:
            return v  # pragma: no cover

        if 'speed_data' not in values:
            return v  # pragma: no cover

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

    def all_points(self) -> List[PointType]:
        return [self.start_point, *self.way_points, self.end_point]

    def point_intervals(self) -> List[Tuple[PointType, PointType]]:
        all_points = self.all_points()
        return list(zip(all_points[:-1], all_points[1:]))


@set_module(MPE_MODULE)
class PathInfo(NamedTuple):
    """Extracted path info
    """

    path: np.ndarray
    start_point: PointType
    end_point: PointType
    travel_time: np.ndarray
    reversed: bool


@set_module(MPE_MODULE)
class ResultPathInfo(ImmutableDataObject):
    """Result path info
    """

    path: np.ndarray
    pieces: Tuple[PathInfo, ...]

    @property
    def point_count(self) -> int:
        return self.path.shape[0]


@overload
def mpe(speed_data: np.ndarray, *,
        start_point: PointType,
        end_point: PointType,
        way_points: WayPointsType = (),
        parameters: Optional['Parameters'] = None) -> ResultPathInfo:
    pass  # pragma: no cover


@overload
def mpe(init_info: InitialInfo, *,
        parameters: Optional['Parameters'] = None) -> ResultPathInfo:
    pass  # pragma: no cover


@set_module(MPE_MODULE)
@singledispatch
def mpe(*args, **kwargs) -> ResultPathInfo:  # noqa
    """Extracts a minimal path

    Usage
    -----

    .. code-block:: python

        mpe(init_info: InitialInfo, *,
            parameters: Optional[Parameters] = None) -> ResultPathInfo

        mpe(speed_data: np.ndarray, *,
            start_point: Sequence[int],
            end_point: Sequence[int],
            way_points: Sequence[Sequence[int]] = (),
            parameters: Optional[Parameters] = None) -> ResultPathInfo

    Returns
    -------
    path_info : ResultPathInfo
        Extracted path info

    """
