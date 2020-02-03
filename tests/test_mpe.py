# -*- coding: utf-8 -*-

import logging
import pytest

from mpe import mpe


@pytest.mark.parametrize('start_point, end_point, point_count', [
    ((37, 255), (484, 300), 752),
])
def test_extract_without_waypoints(caplog, speed_image_retina, start_point, end_point, point_count):
    caplog.set_level(logging.DEBUG)
    path_info = mpe(speed_image_retina, start_point=start_point, end_point=end_point)

    assert path_info.point_count == point_count
