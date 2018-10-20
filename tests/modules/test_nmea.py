#!/usr/bin/env python

'''Test nmea module
'''

import sys
import pytest
import os
import threading, time

if sys.version_info >= (3, 3):
    from unittest import mock
else:
    import mock

#for generating mocked mavlink messages
from pymavlink.dialects.v20 import common

import cuav.modules.nmea as nmea

@pytest.fixture
def mpstate():
    mpstatetmp = mock.Mock() #mpstatedummy.MPState()
    mpstatetmp.public_modules = {}
    mpstatetmp.command_map = {}
    mpstatetmp.completions = {}
    mpstatetmp.completion_functions = {}
    mpstatetmp.multi_instance = {}
    mpstatetmp.instance_count = {}
    return mpstatetmp

def test_load_module(mpstate):
    '''Just initialise the module'''
    loadedModule = nmea.init(mpstate)
    loadedModule.unload()

#def test_module_serial(mpstate):
#    '''try changing module settings via MAVProxy CLI'''
#    loadedModule = nmea.init(mpstate)
#    loadedModule.cmd_nmea([])
#    loadedModule.cmd_nmea(['serial', '/dev/ttyS0', '9600', '8', 'N', '1'])
#    loadedModule.unload()
#
#    assert loadedModule.serial is not None
#    #assert loadedModule.baudrate == 9600

def test_module_udp(mpstate):
    '''try changing module settings via MAVProxy CLI'''
    loadedModule = nmea.init(mpstate)
    loadedModule.cmd_nmea([])
    loadedModule.cmd_nmea(['udp', '127.0.0.1:17650'])
    loadedModule.unload()

    assert loadedModule.udp_output_port is not None
    #assert loadedModule.baudrate == 9600
    
def test_formats(mpstate):
    '''Test the various formatting functions'''
    loadedModule = nmea.init(mpstate)

    fmtdate = loadedModule.format_date(1525594631)
    fmttime = loadedModule.format_time(1525594631)
    fmtlat = loadedModule.format_lat(34.56)
    fmtlong = loadedModule.format_lon(-134.7)
    loadedModule.unload()

    assert fmtdate == "060518"
    assert fmttime == "081711.000"
    assert fmtlat == "3433.60000,N"
    assert fmtlong == "13442.00000,W"

def test_messages(mpstate):
    '''Test the various gps messages'''
    loadedModule = nmea.init(mpstate)

    ggamsg = loadedModule.format_gga(1525594631, 10, -45, 4, 12, 1.2, 120)
    rmcmsg = loadedModule.format_rmc(1525594631, 4, -20, 0, 1.3, 187)
    loadedModule.unload()

    assert ggamsg == "$GPGGA,081711.000,1000.00000,N,04500.00000,W,4,12,01.2,0120.00,M,0041.20,M,,*41"
    assert rmcmsg == "$GPRMC,081711.000,4,2000.00000,S,00000.00000,E,1.30,187.00,060518,,*61"

def test_mavlink(mpstate):
    '''Send some mavlink packets through'''
    loadedModule = nmea.init(mpstate)

    #mock some packets
    loadedModule.nmea_settings.target_system = 0
    msgGPSINT = common.MAVLink_gps_raw_int_message(1525594631, 4, 230000000, -344500000, 100000, 3, 2, 3400, 2300, 12)
    msgGPSPOS = common.MAVLink_global_position_int_message(2000, 230100000, -344560000, 110000, 100000, 3000, 200, 1, 9000)

    loadedModule.mavlink_packet(msgGPSINT)
    loadedModule.mavlink_packet(msgGPSPOS)
    loadedModule.unload()

    assert loadedModule.num_sat == 12
    assert loadedModule.hdop == 0.03
    assert loadedModule.fix_quality == 1
    assert loadedModule.last_time_boot_ms == 2000
