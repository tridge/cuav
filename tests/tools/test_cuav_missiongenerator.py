#!/usr/bin/env python

'''Test cuav mission generator (OBC 2014)
'''

import sys
import pytest
import os
import cuav.tools.cuav_missiongenerator as missiongenerator

def test_do_missiongen():
    missionfile = os.path.join('.', 'tests', 'testdata', 'OBC Waypoints 2014.kml')
    
    gen = missiongenerator.MissionGenerator(missionfile)
    gen.Process('SA-', 'MB-')
    gen.CreateEntryExitPoints('EL-01,EL-02', 'EL-03,EL-04')
    groundWidth = gen.getCameraWidth(100, 0.098, 1280)
    assert groundWidth == 125.44
    
    gen.CreateSearchPattern(width = groundWidth, overlap=60, offset=150, wobble=10, alt=100)
    
    assert gen.getPolygonLength() == 70600

