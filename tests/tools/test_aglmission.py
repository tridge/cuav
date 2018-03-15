#!/usr/bin/env python

'''Test OpenCV speedtest
'''

import sys
import pytest
import os
import cuav.tools.agl_mission as agl_mission

def test_do_aglmission():
    missionfile = os.path.join('.', 'tests', 'testdata', 'cmac-image-wp.txt')
    outfile = os.path.join('.', 'tests', 'testdata', 'cmac-image-wp-new.txt')
    agl = 90
    speed = 25
    wp = agl_mission.fix_alt(missionfile, agl, None, outfile)
    wp = agl_mission.add_points(wp, agl, 50, 50, 100)
    wp = agl_mission.fix_climb(wp, speed, 3)
    agl_mission.report_points(wp, speed)
    assert os.path.isfile(outfile) 
    os.remove(outfile)

