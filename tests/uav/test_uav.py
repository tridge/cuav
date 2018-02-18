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
    
    assert l_w[0] == 499.99999999999989
    assert l_w[1] == 1000
    assert l_w[2] == -600
    assert l_w[3] == 1
    assert scale == 99.003328892062072

def test_rotationMatrix():
    Rc = rotationMatrix(45, 60, 30)
    
    assert Rc.shape == (3, 3)
    assert Rc[0,0] == -0.14691108312079304
    assert Rc[1,1] == 0.33729193922741485
    assert Rc[2,2] == -0.50032348104751134
    
