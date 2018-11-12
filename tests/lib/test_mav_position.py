#!/usr/bin/env python
'''
tests for mav_position.py
'''

import sys, os, time, random, functools, math, datetime, fractions
import pytest
from cuav.lib import mav_position
            

def test_MavPosition():
    mav = mav_position.MavPosition(-30, 145, 34.56, 20, -56.67, 345, frame_time=None)
    assert str(mav) == "MavPosition(pos -30.000000 145.000000 alt=34.6 roll=20.0 pitch=-56.7 yaw=345.0)"

def test_MavInterpolator():
    mpos = mav_position.MavInterpolator(gps_lag=0)
    mpos.set_logfile(os.path.join(os.getcwd(), 'tests', 'testdata', 'flight.tlog'))
    #frame_time = datetime.datetime.strptime('2016111223465695Z', "%Y%m%d%H%M%S%fZ")
    #frame_time = mav_position.datetime_to_float(frame_time)
    try:
        pos = mpos.position(1478994416.95, 0,roll=None)
    except mav_position.MavInterpolatorException as e:
        assert str(e) == "no msgs of type GLOBAL_POSITION_INT before Sun Nov 13 10:46:56 2016 last=" or "no msgs of type GLOBAL_POSITION_INT before Sun Nov 13 23:46:56 2016 last="

    pos = mpos.position(1478996416.95, 0,roll=None)
    assert pos.lat == -35.3654064 and pos.lon == 149.1643571
    
    poss = mpos.position(1478996416.00, 0,roll=None)
    assert poss.lat == -35.3654064 and poss.lon == 149.1643571
    
    posss = mpos.position(1478998416.34, 0,roll=None)
    assert posss.lat == -35.3654064 and posss.lon == 149.1643571

def test_Fraction():
    fr = mav_position.Fraction(0.3)
    assert fr == fractions.Fraction(3, 10)
    
def test_dms_to_decimal():
    assert mav_position.dms_to_decimal((10, 1), (10, 1), (10, 1)) == 10.169444444444444
    assert mav_position.dms_to_decimal((8, 1), (9, 1), (10, 200), b'S') == -8.15001388888889
    
def test_decimal_to_dms():
    assert mav_position.decimal_to_dms(50.445891) == [(50, 1), (26, 1), (113019, 2500)]
    assert mav_position.decimal_to_dms(-125.976893) == [(125, 1), (58, 1), (92037, 2500)]

def test_exif_position():
    testfile = os.path.join(os.getcwd(), 'tests', 'testdata', 'exifimg2018021100040450Z.jpg')
    testfiletwo = os.path.join(os.getcwd(), 'tests', 'testdata', 'exifimg2018021100041320Z.jpg')
    pos = mav_position.exif_position(testfile)
    postwo = mav_position.exif_position(testfiletwo)
    #Note all altitudes are AGL
    assert pos.lat == -35.36327931731889 and pos.lon == 149.16453320132894
    assert pos.altitude == 93.741455078125 and pos.time == 1518307444.5
    assert pos.yaw == 0
    assert postwo.lat == -35.36223437803768 and postwo.lon == 149.1643464050329
    assert postwo.altitude == 83.53874206542969 and postwo.time == 1518307453.2
    assert postwo.yaw == 351.7056906395551
    
def test_KmlPosition():
    testfile = os.path.join(os.getcwd(), 'tests', 'testdata', 'OBC Waypoints.kml')
    kmzpos = mav_position.KmlPosition(testfile)
    pos = kmzpos.position("MB-4")
    assert pos.lat == -26.577306 and pos.lon == 151.8403333333334
    assert pos.altitude == 0 and pos.time == 0
    assert pos.yaw == 0

@pytest.mark.skipif(sys.platform.startswith("win"), reason="SRTM caching broken in Windows")
def test_TriggerPosition():
    testfile = os.path.join(os.getcwd(), 'tests', 'testdata', 'robota.trigger')
    testpng = os.path.join(os.getcwd(), 'tests', 'testdata', 'exif2016111223464876Z.png')
    tpos = mav_position.TriggerPosition(testfile)
    pos = tpos.position(testpng)
    #Note all altitudes are AGL
    assert pos.lat == -20.1234 and pos.lon == 145.456
    assert pos.altitude == 10 and pos.time == 1478994363.0
    assert pos.yaw == 0

@pytest.mark.skipif(sys.platform.startswith("win"), reason="SRTM caching broken in Windows")
def test_get_ground_alt():
    assert mav_position.get_ground_alt(-35, 149) == 596.8509306748258
