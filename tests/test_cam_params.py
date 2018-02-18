#!/usr/bin/env python

'''Test Camera params
'''

from numpy import array
import json
import sys
import pytest
import os
from cuav.camera.cam_params import CameraParams

def test_cam_params_txt():
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1280, yresolution=960)
    C.save('foo.txt')
    C2 = CameraParams.fromfile('foo.txt')
    assert str(C) == str(C2)
    os.remove('foo.txt')

def test_cam_params_dict():
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=3000, yresolution=4000)
    Cdict = C.todict()
    C2dict = CameraParams.fromdict(Cdict)
    assert C.todict() == C2dict.todict()

