#!/usr/bin/env python

'''Test camera calibration
'''

import sys
import pytest
import os
import cuav.tools.calibrate as calibrate

from cuav.camera.cam_params import CameraParams

def test_do_calibrate():
    dirry = os.path.join('.', '.', 'cuav', 'data', 'calibration_images_2014')
    outfile = os.path.join('.', '.', 'cuav', 'data', 'calibration_images_2014', 'params.json')
    calibrate.calibrate(dirry)
    assert os.path.isfile(outfile)
    C = CameraParams.fromfile(outfile)
    assert C is not None
    os.remove(outfile)

