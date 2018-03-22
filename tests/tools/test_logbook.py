#!/usr/bin/env python

'''Test OpenCV speedtest
'''

import sys
import pytest
import os
import cuav.tools.logbook as logbook

def test_do_logbook():
    logfile = os.path.join(os.getcwd(), 'tests', 'testdata', 'flight.tlog')
    logbookentries = []
    total_time = 0.0
    total_dist = 0.0
    
    for f in [logfile, logfile]:
        (ftime, fdist) = logbook.flight_time(logbookentries, f, None, 1, 1)
        total_time += ftime
        total_dist += fdist
    assert total_time == 162
    assert total_dist == 2614.8674443951145


