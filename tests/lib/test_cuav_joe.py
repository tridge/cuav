#!/usr/bin/env python
'''
test program for cuav_joe
'''

import sys, os, time, random, functools
import pytest
from cuav.lib import cuav_joe, cuav_region, mav_position
from cuav.camera.cam_params import CameraParams


def test_JoeLog_JoeIterator():
    joelog = cuav_joe.JoeLog(os.path.join('.', 'joe.log'), False)
    regions = []
    frame_time = 1478954763.0
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1024, yresolution=768)
    regions.append(cuav_region.Region(10, 10, 25, 23, None, scan_score=450))
    regions.append(cuav_region.Region(200, 205, 252, 236, None, scan_score=420))
    pos = mav_position.MavPosition(-30, 145, 34.56, 20, -56.67, 345, frame_time)
    joelog.add_regions(frame_time, regions, pos, 'img2017111312451230Z.png')
    
    joeread = cuav_joe.JoeIterator(os.path.join('.', 'joe.log'))
    
    joeret = joeread.getjoes()

    assert len(joeret) == 2
    assert joeret[0] == "JoePosition(lat=-30.000235 lon=144.999639 MavPosition(pos -30.000000 145.000000 alt=34.6 roll=20.0 pitch=-56.7 yaw=345.0) img2017111312451230Z.png None (10, 10, 25, 23) latlon=(-30.000235126851315, 144.9996388367703) score=None Sat Nov 12 23:46:03 2016 2016111212460300Z)" or "JoePosition(lat=-30.000235 lon=144.999639 MavPosition(pos -30.000000 145.000000 alt=34.6 roll=20.0 pitch=-56.7 yaw=345.0) img2017111312451230Z.png None (10, 10, 25, 23) latlon=(-30.000235126851315, 144.9996388367703) score=None Sat Nov 12 12:46:03 2016 2016111212460300Z)"
    assert joeret[1] == "JoePosition(lat=-30.000367 lon=144.999711 MavPosition(pos -30.000000 145.000000 alt=34.6 roll=20.0 pitch=-56.7 yaw=345.0) img2017111312451230Z.png None (200, 205, 252, 236) latlon=(-30.000366794010567, 144.9997107272955) score=None Sat Nov 12 23:46:03 2016 2016111212460300Z)" or "JoePosition(lat=-30.000367 lon=144.999711 MavPosition(pos -30.000000 145.000000 alt=34.6 roll=20.0 pitch=-56.7 yaw=345.0) img2017111312451230Z.png None (200, 205, 252, 236) latlon=(-30.000366794010567, 144.9997107272955) score=None Sat Nov 12 12:46:03 2016 2016111212460300Z)"
    
    os.remove(os.path.join('.', 'joe.log'))
    
