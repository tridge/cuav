#!/usr/bin/env python

'''Test cuav thermal view
'''

import sys
import pytest
import os
import cuav.tools.thermal_view as thermal_view

def test_convert_image():
    infile = os.path.join('.', 'tests', 'testdata', 'raw2016111223465120Z.png')
    outfile = thermal_view.convert_image(infile, 5600, 0.75, 0.4)
    assert outfile is not None

def test_show_value():
    infile = os.path.join('.', 'tests', 'testdata', 'raw2016111223465120Z.png')
    thermal_view.show_value(50, 67, infile)

def test_file_list():
    infolder = os.path.join('.', 'tests', 'testdata')
    files = thermal_view.file_list(infolder, ['jpg', 'jpeg', 'png'])
    assert len(files) == 6


