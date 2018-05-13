#!/usr/bin/env python

'''Test camera calibration
'''

import sys
import pytest
import os
import cuav.tools.calibrate as calibrate

from cuav.camera.cam_params import CameraParams

def test_do_calibrate():
    dirry = os.path.join('.', 'cuav', 'data', 'ChameleonArecort')
    outfile = os.path.join('.', 'cuav', 'data', 'ChameleonArecort', 'paramsout.json')
    calibrate.calibrate(dirry, 10, 7)
    assert os.path.isfile(outfile)
    C = CameraParams.fromfile(outfile)
    assert C is not None
    os.remove(outfile)

