#!/usr/bin/env python

'''Test Geo conversions
'''

import sys
import pytest
import os
from cuav.uav.geo import geodetic

def test_computeZoneAndBand():
    lat = -( 37.0 + 39.0/60.0 + 10.15610/3600.0)
    lon = +(143.0 + 55.0/60.0 + 35.38390/3600.0)
    g = geodetic()
    (zone, band) = g.computeZoneAndBand(lat, lon)
    assert zone == 54
    assert band == 'H'

def test_geoToGrid():
    lat = -( 37.0 + 39.0/60.0 + 10.15610/3600.0)
    lon = +(143.0 + 55.0/60.0 + 35.38390/3600.0)
    g = geodetic()
    (zone, band) = g.computeZoneAndBand(lat, lon)
    (northing, easting) = g.geoToGrid(lat, lon, zone, band)
    assert northing == 5828674.3400593391
    assert easting == 758173.79728005431
    
def test_gridToGeo():
    northing = 5796489.777
    easting = 273741.297
    band = 'A'
    zone = 55
    g = geodetic()
    (lat_, lon_) = g.gridToGeo(northing, easting, zone, band)
    assert lat_ == -37.951033415544593
    assert lon_ == 144.42486789295646

