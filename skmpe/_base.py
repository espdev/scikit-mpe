# -*- coding: utf-8 -*-

import collections
import functools
import inspect
import logging
from typing import Tuple, Sequence, List, NamedTuple, TYPE_CHECKING

from pydantic import BaseModel, Extra, validator, root_validator
import numpy as np

if TYPE_CHECKING:
    from ._parameters import Parameters  # noqa
    from ._mpe import PathExtractionResult  # noqa


PointType = Sequence[int]
FloatPointType = Sequence[float]
PointSequenceType = Sequence[PointType]

InitialPointType = Tuple[int, ...]
InitialWayPointsType = Tuple[InitialPointType, ...]


MPE_MODULE = 'skmpe'
MIN_NDIM = 2


logger = logging.getLogger(MPE_MODULE)
logger.addHandler(logging.NullHandler())


def mpe_module(obj):
    """Replace __module__ for decorated object

    Returns
    -------
    obj : Any
        The object with replaced __module__
    """

    obj.__module__ = MPE_MODULE
    return obj


def singledispatch(func_stub):
    """Single dispatch wrapper with default implementation that raises the exception for invalid signatures
    """

    @functools.wraps(func_stub)
    @functools.singledispatch
    def _dispatch(*args, **kwargs):
        sign_args = ', '.join(f'{type(arg).__name__}' for arg in args)
        sign_kwargs = ', '.join(f'{kwname}: {type(kwvalue).__name__}' for kwname, kwvalue in kwargs.items())
        allowed_signs = ''
        i = 0

        for arg_type, func in _dispatch.registry.items():
            if arg_type is object:
                continue
            i += 1
            sign = inspect.signature(func)
            allowed_signs += f'  [{i}] => {sign}\n'

        raise TypeError(
            f"call '{func_stub.__name__}' with invalid signature:\n"
            f"  => ({sign_args}, {sign_kwargs})\n\n"
            f"allowed signatures:\n{allowed_signs}")

    return _dispatch


@mpe_module
class ImmutableDataObject(BaseModel):
    """Base immutable data object with validating fields
    """

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_mutation = False


@mpe_module
class InitialInfo(ImmutableDataObject):
    """Initial info data model

    .. py:attribute:: speed_data

        Speed data in numpy ndarray

    .. py:attribute:: start_point

        The starting point

    .. py:attribute:: end_point

        The ending point

    .. py:attribute:: way_points

        The tuple of way points

    """

    speed_data: np.ndarray

    start_point: InitialPointType
    end_point: InitialPointType
    way_points: InitialWayPointsType = ()

    @validator('start_point', 'end_point', 'way_points', pre=True)
    def _to_canonical(cls, v):
        return np.asarray(v).tolist()

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
        """Returns all initial points"""
        return [self.start_point, *self.way_points, self.end_point]

    def point_intervals(self) -> List[Tuple[PointType, PointType]]:
        """Returns the list of the tuples of initial point intervals"""
        all_points = self.all_points()
        return list(zip(all_points[:-1], all_points[1:]))


@mpe_module
class PathInfo(NamedTuple):
    """The named tuple with info about extracted path or piece of path

    .. py:attribute:: path

        The path in numpy ndarray

    .. py:attribute:: start_point

        The starting point

    .. py:attribute:: end_point

        The ending point

    .. py:attribute:: travel_time

        The travel time numpy ndarray

    .. py:attribute:: extraction_result

        The path extraction result in :class:`PathExtractionResult` that is returned by :class:`MinimalPathExtractor`

    .. py:attribute:: reversed

        The flag is true if the extracted path is reversed

    """

    path: np.ndarray
    start_point: PointType
    end_point: PointType
    travel_time: np.ndarray
    extraction_result: 'PathExtractionResult'
    reversed: bool


@mpe_module
class PathInfoResult(NamedTuple):
    """The named tuple with path info result

    .. py:attribute:: path

        Path data in numpy array

    .. py:attribute:: pieces

        The tuple of :class:`PathInfo` for every path piece
    """

    path: np.ndarray
    pieces: Tuple[PathInfo, ...]

    @property
    def point_count(self) -> int:
        """Returns the number of path points"""
        return self.path.shape[0]
