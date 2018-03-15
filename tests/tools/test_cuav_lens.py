#!/usr/bin/env python

'''Test lens equations
'''

import sys
import pytest
import os
import cuav.tools.cuav_lens as cuav_lens


def test_aov():
    assert cuav_lens.aov(5, 2.8) == 83.52059940779574
    
def test_groundwidth():
    assert cuav_lens.groundwidth(122, 5, 2.8) == 217.85714285714286

def test_pixelwidth():
    assert cuav_lens.pixelwidth(1280, 122, 5, 2.8) == 0.17020089285714285

def test_pixelarea():
    assert cuav_lens.pixelarea(1280, 122, 5, 2.8) == 0.02896834392936862

def test_lamparea():
    assert cuav_lens.lamparea(6.5) == 0.0033183072403542195

def test_lamppower():
    assert cuav_lens.lamppower(10, 50) == 5
    
def test_lamppixelpower():
    assert cuav_lens.lamppixelpower(10, 50, 1280, 6.5, 122, 5, 2.8) == 5
    
def test_sunonlamp():
    assert cuav_lens.sunonlamp(1500, 6.5) == 4.9774608605313295
    
def test_sunreflected():
    assert cuav_lens.sunreflected(0.2, 1500, 1280, 122, 5, 2.8) == 8.690503178810586

def test_apparentbrightness():
    assert cuav_lens.apparentbrightness(10, 50, 1280, 6.5, 1.0, 0.2, 1500, 122, 5, 2.8) == 1.575340679028935

