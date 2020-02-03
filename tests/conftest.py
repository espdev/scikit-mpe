# -*- coding: utf-8 -*-

import pytest

from skimage.data import retina
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.filters import sato
from skimage.exposure import adjust_log


@pytest.fixture(scope='session')
def speed_image_retina():
    image_data = rescale(rgb2gray(retina())[260:1280, 90:800], 0.7)
    speed_data = adjust_log(sato(image_data), gain=5.)

    return speed_data
