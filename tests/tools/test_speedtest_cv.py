#!/usr/bin/env python

'''Test OpenCV speedtest
'''

import sys
import pytest
import os
import cuav.tools.speedtest_cv as speedtest_cv

def test_do_speedtest():
    speedtest_cv.do_speedtest(os.path.join('.', 'tests', 'testdata', 'raw2016111223465120Z.png'))

def test_circle_highest():
    speedtest_cv.circle_highest(os.path.join('.', 'tests', 'testdata', 'raw2016111223465120Z.png'))


