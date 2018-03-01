#!/usr/bin/env python
'''
test program for cuav_mosiac
'''

import sys, os, time, random, functools, cv2
import pytest
import numpy as np
from cuav.lib import cuav_region, cuav_util
from cuav.lib import mav_position
from cuav.lib import cuav_mosaic


def test_MosaicRegion():
    regOne = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
    regOne.latlon=(-26.6398870, 151.8220000)
    regOne.score = 20
    pos = mav_position.MavPosition(-30, 145, 34.56, 20, -56.67, 345, frame_time=None)
    pos.time = int(time.time())
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    full_thumbnail = im_orig[1:80, 1:80]
    small_thumbnail = im_orig[2:12, 10:20]
    mos = cuav_mosaic.MosaicRegion(2, regOne, "nofile.png", pos, full_thumbnail, small_thumbnail)
    mos.tag_image_available()
    assert mos.small_thumbnail[2,9,0] == 0
    assert mos.small_thumbnail[2,9,1] == 255
    assert mos.small_thumbnail[2,9,2] == 255
    assert "MavPosition(pos -30.000000 145.000000 alt=34.6 roll=20.0 pitch=-56.7 yaw=345.0)" in str(mos)

def test_MosaicImage():
    pos = mav_position.MavPosition(-30, 145, 34.56, 20, -56.67, 345, frame_time=None)
    pos.time = int(time.time())
    im = cuav_mosaic.MosaicImage(int(time.time()), "nofile.png", pos)
    assert "nofile.png" in str(im)
    
