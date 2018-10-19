#!/usr/bin/env python
'''
test program for cuav_landingregion
'''

import sys, os, time, random, functools
import pytest
import numpy as np
from cuav.lib import cuav_region, cuav_landingregion, mav_position



def test_addLandingZone():
    lz = cuav_landingregion.LandingZone()
    for i in range(0, 10):
        r = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
        r.latlon = (23, 34)
        pos = mav_position.MavPosition(23, 24, 80, 0, 0, 0, 1)
        r.score = 20
        lz.checkaddregion(r, pos)

    assert len(lz.regions) == 10

def test_addLandingZoneMany():
    lz = cuav_landingregion.LandingZone()
    for i in range(0, 100):
        r = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
        r.latlon = (random.uniform(-90, 90), random.uniform(-180, 180))
        r.score = random.randint(0, 1000)
        pos = mav_position.MavPosition(r.latlon[0], r.latlon[1], 80, 0, 0, 0, 1)
        lz.checkaddregion(r, pos)

    assert len(lz.regions) == 100

def test_averagepos():
    lz = cuav_landingregion.LandingZone()
    rg = []
    for i in range(0, 100):
        r = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
        r.latlon = (random.uniform(-90, 90), random.uniform(-180, 180))
        r.score = random.randint(0, 1000)
        pos = mav_position.MavPosition(r.latlon[0], r.latlon[1], 80, 0, 0, 0, 1)
        rg.append(r)
        
    ret = lz.average_pos(rg)
    assert ret[0] > -90 and ret[0] < 90
    assert ret[1] > -180 and ret[1] < 180
    
def test_calcLandingZone():
    lz = cuav_landingregion.LandingZone()
    for i in range(0, 100):
        r = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
        r.latlon = (random.uniform(-0.001, 0.001)+34, random.uniform(-0.001, 0.001)-140)
        r.score = random.randint(0, 1000)
        pos = mav_position.MavPosition(r.latlon[0], r.latlon[1], 80, 0, 0, 0, 1)
        lz.checkaddregion(r, pos)

    ret = lz.calclandingzone()
    assert ret.latlon[0] > -90 and ret.latlon[0] < 90
    assert ret.latlon[1] > -180 and ret.latlon[1] < 180
    assert ret.maxrange > 0
    assert ret.avgscore > 0
    assert ret.numregions > 0


