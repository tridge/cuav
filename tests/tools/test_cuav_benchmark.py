#!/usr/bin/env python

'''Test cuav benchmark
'''

import sys
import pytest
import os
import cuav.tools.cuav_benchmark as cuav_benchmark


def test_cuav_benchmark():
    infile = os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png')
    cuav_benchmark.process(infile, 10)

