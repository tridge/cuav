#!/usr/bin/env python
'''
CUAV module for on companion computer
Andrew Tridgell
'''

from MAVProxy.modules.lib import mp_module
from pymavlink import mavutil
import time, math

# LED states
LED_OFF=(0,0,'OFF')
LED_RED=(0,1,'RED')
LED_GREEN=(1,0,'GREEN')
LED_FLASH=(1,1,'FLASH')

class CUAVCompanionModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CUAVCompanionModule, self).__init__(mpstate, "CUAV", "CUAV companion")
        self.led_state = LED_OFF
        self.led_force = None
        self.led_send_time = 0
        self.button_change_time = 0
        self.last_attitude_ms = 0
        self.add_command('cuavled', self.cmd_cuavled, "cuav led command", ['<red|green|flash|off|refresh>'])

    def cmd_cuavled(self, args):
        '''handle cuavled commands'''
        usage = "usage: cuavled red|green|flash|off|refresh"
        if len(args) == 0:
            print(usage)
            return
        if args[0] == 'red':
            self.force_leds(LED_RED)
        elif args[0] == 'green':
            self.force_leds(LED_GREEN)
        elif args[0] == 'flash':
            self.force_leds(LED_FLASH)
        elif args[0] == 'off':
            self.force_leds(LED_OFF)
        elif args[0] == 'refresh':
            self.led_force = None
            self.led_state = None
            self.update_led_state()
            
    def force_leds(self, state):
        self.led_force = state
        self.set_leds(state)
            

    def set_relay(self, relaynum, value):
        self.master.mav.command_long_send(self.target_system,
                                          self.target_component,
                                          mavutil.mavlink.MAV_CMD_DO_SET_RELAY, 0,
                                          relaynum, value,
                                          0, 0, 0, 0, 0)
        

    def set_leds(self, state):
        '''set two LEDs via relays'''
        self.ack_wait = 2
        self.led_state = state
        self.led_send_time = time.time()
        self.set_relay(0, state[0])
        self.set_relay(1, state[1])
        if state == LED_FLASH:
            # also play warning tune
            self.master.mav.play_tune_send(self.settings.target_system,
                                           self.settings.target_component,
                                           'AAAAAA')

    def idle_task(self):
        '''run periodic tasks'''
        pass

    def update_led_state(self):
        '''update LED state'''
        if self.led_force is not None:
            led_state = self.led_force
        elif self.master.motors_armed():
            led_state = LED_RED
        elif time.time() - self.button_change_time < 60:
            led_state = LED_FLASH
        else:
            led_state = LED_GREEN
        if led_state != self.led_state:
            print("Changing LEDs to: %s" % led_state[2])
            self.set_leds(led_state)

    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        now = time.time()
        if m.get_type() == "BUTTON_CHANGE":
            time_since = max(m.time_boot_ms - m.last_change_ms,0) * 1.0e-3
            self.button_change_time = time.time() - time_since
            self.update_led_state()
        if m.get_type() == 'HEARTBEAT':
            self.update_led_state()
        if m.get_type() == 'COMMAND_ACK' and m.command == mavutil.mavlink.MAV_CMD_DO_SET_RELAY and self.ack_wait > 0:
            self.ack_wait -= 1
            if self.ack_wait == 0:
                print("LEDs updated: %s" % self.led_state[2])
        if m.get_type() == 'ATTITUDE':
            if m.time_boot_ms < self.last_attitude_ms:
                self.led_state = None
            self.last_attitude_ms = m.time_boot_ms

def init(mpstate):
    '''initialise module'''
    return CUAVCompanionModule(mpstate)
