#!/usr/bin/env python
'''
test program for cuav_landingregion
'''

import sys, os, time, random, functools
import pytest
import numpy as np
from cuav.lib import cuav_region, cuav_landingregion


def test_addLandingZone():
    lz = cuav_landingregion.LandingZone()
    for i in range(0, 10):
        r = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
        r.latlon = (23, 34)
        r.score = 20
        lz.checkaddregion(r)

    assert len(lz.regionClumps) == 1

def test_addLandingZoneMany():
    lz = cuav_landingregion.LandingZone()
    for i in range(0, 100):
        r = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
        r.latlon = (random.uniform(-90, 90), random.uniform(-180, 180))
        r.score = random.randint(0, 1000)
        lz.checkaddregion(r)

    assert len(lz.regionClumps) == 100

def test_calcLandingZone():
    lz = cuav_landingregion.LandingZone()
    for i in range(0, 100):
        r = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
        r.latlon = (random.uniform(-0.001, 0.001)+34, random.uniform(-0.001, 0.001)-140)
        r.score = random.randint(0, 1000)
        lz.checkaddregion(r)

    lz.calclandingzone()
    assert len(lz.regionClumps) < 100 and len(lz.regionClumps) > 0


