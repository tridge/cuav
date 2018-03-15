#!/usr/bin/env python

'''Test Camera params
'''

from numpy import array
import json
import sys
import pytest
import os, numpy
from cuav.camera.cam_params import CameraParams

def test_cam_params_txt():
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1280, yresolution=960)
    C.setParams([[1, 2, 3], [4, 5, 6]], [0, 6, 7, 8])
    C.save('foo.txt')
    C2 = CameraParams.fromfile('foo.txt')
    assert str(C) == str(C2)
    assert numpy.array_equal(C.K, C2.K)
    assert numpy.array_equal(C.D, C2.D)
    os.remove('foo.txt')

def test_cam_params_dict():
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=3000, yresolution=4000)
    C.setParams([[1, 2, 3], [4, 5, 6]], [0, 6, 7, 8])
    Cdict = C.todict()
    C2dict = CameraParams.fromdict(Cdict)
    assert C.todict() == C2dict.todict()
    assert numpy.array_equal(C.K, C2dict.K)
    assert numpy.array_equal(C.D, C2dict.D)

