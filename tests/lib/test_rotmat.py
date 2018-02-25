#!/usr/bin/env python
'''
tests for rotmat.py
'''

import sys, os, time, random, functools, math
import pytest
from cuav.lib.rotmat import Matrix3, Vector3, Plane, Line
            

def test_rotmat_euler():
    '''check that from_euler() and to_euler() are consistent'''
    print("testing euler maths")
    m = Matrix3()
    from math import radians, degrees
    for r in range(-179, 179, 10):
        for p in range(-89, 89, 10):
            for y in range(-179, 179, 10):
                m.from_euler(radians(r), radians(p), radians(y))
                (r2, p2, y2) = m.to_euler()
                v1 = Vector3(r,p,y)
                v2 = Vector3(degrees(r2),degrees(p2),degrees(y2))
                diff = v1 - v2
                assert diff.length() < 1.0e-12


def test_two_vectors():
    '''test the from_two_vectors() method'''
    import random
    for i in range(1000):
        v1 = Vector3(1, 0.2, -3)
        v2 = Vector3(random.uniform(-5,5), random.uniform(-5,5), random.uniform(-5,5))
        m = Matrix3()
        m.from_two_vectors(v1, v2)
        v3 = m * v1
        diff = v3.normalized() - v2.normalized()
        (r, p, y) = m.to_euler()
        assert diff.length() < 0.001


def test_plane():
    '''testing line/plane intersection'''
    print("testing plane/line maths")
    plane = Plane(Vector3(0,0,0), Vector3(0,0,1))
    line = Line(Vector3(0,0,100), Vector3(10, 10, -90))
    p = line.plane_intersection(plane)
    assert abs(11.1111 - p.x) < 0.0001
    assert abs(11.1111 - p.y) < 0.0001
    assert abs(0 - p.z) < 0.0001


