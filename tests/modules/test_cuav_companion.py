#!/usr/bin/env python

'''Testing the cuav_companion module
'''

import sys
import pytest
import os
import mock
import threading, time

#for generating mocked mavlink messages
from pymavlink.dialects.v20 import common

import cuav.modules.cuav_companion as cuav_companion

@pytest.fixture
def mpstate():
    mpstatetmp = mock.Mock() #mpstatedummy.MPState()
    mpstatetmp.public_modules = {}
    mpstatetmp.command_map = {}
    mpstatetmp.completions = {}
    mpstatetmp.completion_functions = {}
    return mpstatetmp

def test_load_module(mpstate):
    '''Just initialise the module'''
    loadedModule = cuav_companion.init(mpstate)
    loadedModule.unload()

def test_module_settings(mpstate):
    '''try changing module settings via MAVProxy CLI'''
    loadedModule = cuav_companion.init(mpstate)
    loadedModule.cmd_cuavled([])
    loadedModule.cmd_cuavled(['red'])
    loadedModule.unload()

    assert loadedModule.led_force == (0,1,'RED')
    assert loadedModule.led_state == (0,1,'RED')

def test_set_leds(mpstate):
    '''change the LED states'''
    loadedModule = cuav_companion.init(mpstate)
    loadedModule.set_leds((1,1,'FLASH'))
    loadedModule.unload()

    assert loadedModule.led_state == (1,1,'FLASH')

def test_mavlink(mpstate):
    '''Send some mavlink packets through'''
    loadedModule = cuav_companion.init(mpstate)
    loadedModule.master.motors_armed.return_value = False
    assert loadedModule.led_state == (0,0,'OFF')

    #mock some packets
    #send HB, assert green
    msgHB = common.MAVLink_heartbeat_message(common.MAV_TYPE_FIXED_WING,
                                             common.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                             common.MAV_MODE_AUTO_DISARMED,
                                             0, common.MAV_STATE_ACTIVE, 2)
    loadedModule.mavlink_packet(msgHB)
    assert loadedModule.led_state == (1,0,'GREEN')

    #send button press, assert flashing
    msgButton = common.MAVLink_button_change_message(10000, 10000, 1)
    loadedModule.mavlink_packet(msgButton)
    assert loadedModule.led_state == (1,1,'FLASH')

    loadedModule.unload()


