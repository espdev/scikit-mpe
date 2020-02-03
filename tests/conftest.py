# -*- coding: utf-8 -*-

import pytest

from skimage.data import retina
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.filters import sato


@pytest.fixture(scope='session')
def speed_image_retina():
    image_data = rescale(rgb2gray(retina())[260:1280, 90:800], 0.5)
    speed_data = sato(image_data)
    return speed_data
