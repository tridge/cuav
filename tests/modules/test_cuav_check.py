#!/usr/bin/env python

'''Testing the cuav_companion module
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
from pymavlink.dialects.v20 import common, ardupilotmega
from pymavlink import mavparm

import cuav.modules.cuav_check as cuav_check
import cuav.modules.camera_ground as cuav_camera_ground

class MPStatusMock(object):
    '''hold status information about the mavproxy'''
    def __init__(self):
        self.logdir = None


class MPMasterMock(object):
    '''hold status information about the mavproxy'''
    def __init__(self):
        self.messages = {}
        
    def __call__(self):
        ret = mock.Mock()
        ret.messages = {}
        return ret

class mstatetmp(object):
    def __init__(self):
        self.public_modules = {}
        self.command_map = {}
        self.completions = {}
        self.completion_functions = {}
        self.map = None
        self.mav_param = mavparm.MAVParmDict()
        self.functions = mock.Mock()
        self.status = MPStatusMock()
        self.status.logdir = os.path.join(os.getcwd(), 'gnd')
        self.master = MPMasterMock()

    def module(self, name):
        '''Find a public module (most modules are private)'''
        if name in self.public_modules:
            return self.public_modules[name]
        return None

    @property
    def console(self):
        return mock.Mock()

    @property
    def settings(self):
        return mock.Mock()

@pytest.fixture
def mpstate():
    return mstatetmp()

def test_load_module(mpstate):
    '''Just initialise the module'''
    loadedModule = cuav_check.init(mpstate)
    loadedModule.unload()

def test_module_settings(mpstate):
    '''try changing module settings via MAVProxy CLI'''
    loadedModule = cuav_check.init(mpstate)
    loadedModule.cmd_cuavcheck([])
    loadedModule.cmd_cuavcheck(['set', 'rpm_threshold', 5000])
    loadedModule.cmd_cuavcheck(['set', 'wind_speed', 3.5])
    loadedModule.cmd_cuavcheck(['set', 'wind_direction', 12])
    loadedModule.unload()

    assert loadedModule.cuav_settings.rpm_threshold == 5000
    assert loadedModule.cuav_settings.wind_speed == 3.5
    assert loadedModule.cuav_settings.wind_direction == 12

def test_toggle_LandingZone(mpstate):
    '''toggle the landing zone on the map'''
    loadedModule = cuav_check.init(mpstate)
    loadedModule.mpstate.public_modules['map'] = mock.Mock()
    loadedModule.mpstate.map = mock.Mock()
    loadedModule.module('map').click_position.return_value = (-32, 145)


    loadedModule.toggle_LandingZone()
    loadedModule.toggle_LandingZone()

    assert loadedModule.showLandingZone == False
    loadedModule.unload()

def test_toggle_JoeZone(mpstate):
    '''toggle the Joe zone on the map'''
    loadedModule = cuav_check.init(mpstate)
    loadedModule.mpstate.public_modules['link'] = mock.Mock()

    loadedModule.toggle_JoeZone()
    assert loadedModule.target == None

    loadedModule.mpstate.public_modules['camera_ground'] = cuav_camera_ground.init(mpstate)
    loadedModule.mpstate.public_modules['camera_ground'].camera_settings.target_radius = 0
    loadedModule.mpstate.map = mock.Mock()
    loadedModule.toggle_JoeZone()
    assert loadedModule.target == None

    loadedModule.module('camera_ground').return_value = True
    loadedModule.module('camera_ground').camera_settings.target_radius = 100
    loadedModule.module('camera_ground').camera_settings.target_latitude = -34
    loadedModule.module('camera_ground').camera_settings.target_longitude = -45
    loadedModule.toggle_JoeZone()
    assert loadedModule.target == (-34, -45, 100)

    loadedModule.unload()

def test_update_button_display(mpstate):
    '''Test the button update display on the console'''
    loadedModule = cuav_check.init(mpstate)

    loadedModule.button_change = common.MAVLink_button_change_message(16000, 1000, 1)
    loadedModule.button_change_recv_time = time.time() - 20
    loadedModule.update_button_display()

    loadedModule.unload()

    assert abs( 25 - loadedModule.button_remaining) < 0.01
    assert loadedModule.button_announce_time > 0

def test_rpm_check(mpstate):
    '''Test the RPM checker'''
    loadedModule = cuav_check.init(mpstate)

    m = ardupilotmega.MAVLink_rpm_message(1500, 0)
    loadedModule.cuav_settings.rpm_threshold = 2000
    loadedModule.last_rpm_announce = time.time() - 40
    loadedModule.rpm_check(m)

    loadedModule.unload()

    assert time.time() - loadedModule.last_rpm_announce < 5

def test_update_airspeed_estimate(mpstate):
    '''Test the airspeed updater'''
    loadedModule = cuav_check.init(mpstate)

    m = common.MAVLink_global_position_int_message(2000, 230100000, -344560000, 110000, 100000, 3000, 200, 1, 9000)
    loadedModule.cuav_settings.wind_speed = 10
    loadedModule.cuav_settings.wind_direction = 88
    loadedModule.update_airspeed_estimate(m)

    loadedModule.unload()

def test_mavlink(mpstate):
    '''Send some mavlink packets through'''
    loadedModule = cuav_check.init(mpstate)
    loadedModule.mpstate.mav_param["ICE_START_CHAN"] = 8
    #loadedModule.master.messages['ATTITUDE'].return_value = common.MAVLink_attitude_message(16500, 160, 40, 1600, 1, 1, 1)

    #mock some packets
    ma = ardupilotmega.MAVLink_rpm_message(1500, 0)
    mb = common.MAVLink_button_change_message(16000, 1000, 1)
    mc = common.MAVLink_rc_channels_scaled_message(16500, 1, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 4)
    md = common.MAVLink_named_value_float_message(16505, 'BAT3VOLT', 5600)

    loadedModule.mavlink_packet(ma)
    loadedModule.mavlink_packet(mb)
    loadedModule.mavlink_packet(mc)

    loadedModule.unload()


