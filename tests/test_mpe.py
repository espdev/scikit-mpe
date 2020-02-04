# -*- coding: utf-8 -*-

import logging
import collections

import pytest

from mpe import mpe, parameters


@pytest.mark.parametrize('start_point, end_point, point_count', [
    ((37, 255), (484, 300), 189),
])
def test_extract_without_waypoints(caplog, retina_speed_image, start_point, end_point, point_count):
    caplog.set_level(logging.DEBUG)
    path_info = mpe(retina_speed_image, start_point=start_point, end_point=end_point)

    assert path_info.point_count == point_count


@pytest.mark.parametrize('start_point, end_point, way_points, ttime_cache, point_count, ttime_count, reversed_count', [
    ((37, 255), (484, 300), ((172, 112), (236, 98), (420, 153)), True, 202, 2, 2),
    ((37, 255), (484, 300), ((172, 112), (236, 98), (420, 153)), False, 201, 4, 0),
])
def test_extract_with_waypoints(caplog, retina_speed_image,
                                start_point, end_point, way_points, ttime_cache,
                                point_count, ttime_count, reversed_count):
    caplog.set_level(logging.DEBUG)

    with parameters(travel_time_cache=ttime_cache):
        path_info = mpe(retina_speed_image,
                        start_point=start_point, end_point=end_point, way_points=way_points)

    assert path_info.point_count == point_count

    ttime_counter = collections.Counter(id(piece.travel_time) for piece in path_info.pieces)
    assert len(ttime_counter) == ttime_count

    assert list(piece.reversed for piece in path_info.pieces).count(True) == reversed_count
