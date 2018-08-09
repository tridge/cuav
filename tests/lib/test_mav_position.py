#!/usr/bin/env python
'''
tests for rotmat.py
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
    assert pos.lat == -35.22515214612007 and pos.lon == 149.53868537397852
    
    poss = mpos.position(1478996416.00, 0,roll=None)
    assert poss.lat == -35.225220995897715 and poss.lon == 149.53850268690633
    
    posss = mpos.position(1478998416.34, 0,roll=None)
    assert posss.lat == -35.07964313826459 and posss.lon == 149.92248564626732

def test_Fraction():
    fr = mav_position.Fraction(0.3)
    assert fr == fractions.Fraction(3, 10)
    
def test_dms_to_decimal():
    assert mav_position.dms_to_decimal(10, 10, 10) == 10.169444444444444
    assert mav_position.dms_to_decimal(8, 9, 10, 'S') == -8.152777777777779
    
def test_decimal_to_dms():
    assert mav_position.decimal_to_dms(50.445891) == [fractions.Fraction(50, 1), fractions.Fraction(26, 1), fractions.Fraction(113019, 2500)]
    assert mav_position.decimal_to_dms(-125.976893) == [fractions.Fraction(125, 1), fractions.Fraction(58, 1), fractions.Fraction(92037, 2500)]
    
def test_exif_position():
    testfile = os.path.join(os.getcwd(), 'tests', 'testdata', 'exif2016111223464876Z.png')
    testfiletwo = os.path.join(os.getcwd(), 'tests', 'testdata', 'exif2016111223465337Z.png')
    pos = mav_position.exif_position(testfile)
    postwo = mav_position.exif_position(testfiletwo)
    #Note all altitudes are AGL
    assert pos.lat == -35.36233013408103 and pos.lon == 149.16527170571393
    assert pos.altitude == 12.202238082998873 and pos.time == 1478994408.76
    assert pos.yaw == 0
    assert postwo.lat == -35.362217222827866 and postwo.lon == 149.16496808821856
    assert postwo.altitude == 22.817455291831344 and postwo.time == 1478994413.37
    assert postwo.yaw == 294.51373509551973
    
def test_KmlPosition():
    testfile = os.path.join(os.getcwd(), 'tests', 'testdata', 'OBC Waypoints.kml')
    kmzpos = mav_position.KmlPosition(testfile)
    pos = kmzpos.position("MB-4")
    assert pos.lat == -26.577306 and pos.lon == 151.8403333333334
    assert pos.altitude == 0 and pos.time == 0
    assert pos.yaw == 0
    
def test_TriggerPosition():
    testfile = os.path.join(os.getcwd(), 'tests', 'testdata', 'robota.trigger')
    testpng = os.path.join(os.getcwd(), 'tests', 'testdata', 'exif2016111223464876Z.png')
    tpos = mav_position.TriggerPosition(testfile)
    pos = tpos.position(testpng)
    #Note all altitudes are AGL
    assert pos.lat == -20.1234 and pos.lon == 145.456
    assert pos.altitude == 10 and (pos.time == 1478954763.0 or pos.time == 1478994363.0)
    assert pos.yaw == 0
    
def test_get_ground_alt():
    assert mav_position.get_ground_alt(-35, 149) == 596.8509306748258
