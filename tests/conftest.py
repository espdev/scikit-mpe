# -*- coding: utf-8 -*-

import warnings
import pytest

from skimage.data import retina
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.filters import sato


@pytest.fixture(scope='session')
def retina_speed_image():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        image_data = rescale(rgb2gray(retina())[260:1280, 90:800], 0.5)
        speed_data = sato(image_data)
    return speed_data
