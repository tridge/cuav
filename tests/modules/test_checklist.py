#!/usr/bin/env python

'''Testing the checklist module (and the libchecklist library)
'''

import sys
import pytest
import os
import mock
import threading, time

#for generating mocked mavlink messages
from pymavlink.dialects.v20 import common
from pymavlink import mavparm

import cuav.modules.checklist as checklist

class mstatetmp(object):
    def __init__(self):
        self.public_modules = {}
        self.command_map = {}
        self.completions = {}
        self.completion_functions = {}
        self.map = None
        self.mav_param = mavparm.MAVParmDict()
        self.functions = mock.Mock()
        self.status = mock.Mock()

    def module(self, name):
        '''Find a public module (most modules are private)'''
        if name in self.public_modules:
            return self.public_modules[name]
        return None

    @property
    def console(self):
        return mock.Mock()

    @property
    def master(self):
        return mock.Mock()

    @property
    def settings(self):
        return mock.Mock()

@pytest.fixture
def mpstate():
    return mstatetmp()

def test_load_module(mpstate):
    '''Just initialise the module'''
    loadedModule = checklist.init(mpstate)
    loadedModule.unload()

def test_mavlink(mpstate):
    '''Send some mavlink packets through'''
    loadedModule = checklist.init(mpstate)
    loadedModule.mpstate.mav_param["RC1_TRIM"] = 0
    loadedModule.mpstate.mav_param["RC2_TRIM"] = 0
    loadedModule.mpstate.mav_param["RC3_TRIM"] = 0
    loadedModule.mpstate.mav_param["RC4_TRIM"] = 0
    loadedModule.mpstate.public_modules['wp'] = mock.Mock()

    # mock some packets
    ma = common.MAVLink_vfr_hud_message(airspeed=21, groundspeed=15, heading=90, throttle=80, alt=30, climb=3.2)
    mb = common.MAVLink_attitude_message(16500, 300, 40, 0, 2, 3, 4)
    mc = common.MAVLink_sys_status_message(1, 1, 1, 1, 5400, 100, 300, 0, 0, 0, 0, 0, 0)
    md = common.MAVLink_heartbeat_message(common.MAV_TYPE_FIXED_WING,
                                             common.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                             common.MAV_MODE_AUTO_DISARMED,
                                             0, common.MAV_STATE_ACTIVE, 2)
    #me = common.MAVLink_gps_raw_int_message(1525594631, 4, 230000000, -344500000, 100000, 3, 2, 3400, 2300, 12)

    loadedModule.mavlink_packet(ma)
    loadedModule.mavlink_packet(mb)
    loadedModule.mavlink_packet(mc)
    loadedModule.mavlink_packet(md)
    #loadedModule.mavlink_packet(me)

    loadedModule.unload()


