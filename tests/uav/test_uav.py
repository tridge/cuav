#!/usr/bin/env python

'''Test uav module
'''

import sys
import pytest
import os
from numpy import sin, cos, pi
from numpy import array
from cuav.uav.uav import uavxfer
from cuav.uav.uav import rotationMatrix

def test_uavxfer():
    xfer = uavxfer()
    xfer.setCameraParams(200.0, 200.0, 512, 480)
    xfer.setCameraOrientation(0.0, 0.0, -pi/2)
    xfer.setPlatformPose(500.0, 1000.0, -700.0, 0.1, -0.1, 0.1)
    
    p_w = array([500. +00., 1000. -00., -600.0])
    p_p = xfer.worldToPlatform(p_w[0], p_w[1], p_w[2])
    p_i = xfer.worldToImage(p_w[0], p_w[1], p_w[2])
    
    (l_w, scale) = xfer.imageToWorld(p_i[0], p_i[1])
    
    assert abs(500 - l_w[0]) < 0.01
    assert abs(1000 - l_w[1]) < 0.01
    assert abs(-600 - l_w[2]) < 0.01
    assert abs(1 - l_w[3]) < 0.01
    assert abs(99 - scale) < 0.01

def test_rotationMatrix():
    Rc = rotationMatrix(45, 60, 30)
    
    assert Rc.shape == (3, 3)
    assert abs(-0.1469110 - Rc[0,0]) < 0.001
    assert abs(0.3372919 - Rc[1,1]) < 0.001
    assert abs(-0.5003234 - Rc[2,2]) < 0.001
    
