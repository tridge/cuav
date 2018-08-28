#!/usr/bin/env python
'''
CUAV module for on companion computer
Andrew Tridgell
'''

from MAVProxy.modules.lib import mp_module
from pymavlink import mavutil
import time, math
from cuav.lib import cuav_util

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
        self.last_mission_check_ms = 0
        self.add_command('cuavled', self.cmd_cuavled, "cuav led command", ['<red|green|flash|off|refresh>'])
        from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
        self.cuav_settings = MPSettings(
            [ MPSetting('wp_center', int, 0, 'center search wp'),
              MPSetting('wp_start', int, 0, 'start search wp'),
              MPSetting('wp_end', int, 0, 'end search wp'),
              MPSetting('wp_land',int, 0, 'landing start wp') ])
        self.add_command('cuav', self.cmd_cuav,
                         'cuav companion control',
                         ['set (CUAVSETTING)'])
        self.add_completion_function('(CUAVSETTING)', self.cuav_settings.completion)
        self.last_wp_move = 0
        self.wp_move_count = 0
        self.last_lz_latlon = None
        self.last_wp_list = None
        self.started_landing = False

    def cmd_cuav(self, args):
        '''handle cuav commands'''
        if len(args) < 1:
            print("usage: cuav set ...")
            return
        elif args[0] == "set":
            self.cuav_settings.command(args[1:])
        
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

    def update_mission(self):
        '''update mission status'''
        if (self.cuav_settings.wp_center == 0 or
            self.cuav_settings.wp_start == 0 or
            self.cuav_settings.wp_end == 0):
            # not configured
            return
        if self.started_landing:
            # no more to do
            return

        # run every 5 seconds
        if self.last_attitude_ms - self.last_mission_check_ms < 5000:
            return
        self.last_mission_check_ms = self.last_attitude_ms
        if self.last_attitude_ms - self.last_wp_move < 2*60*1000:
            # only move waypoints every 2 minutes
            return
        try:
            wpmod = self.module('wp')
            cam = self.module('camera_air') 
            lz = cam.lz
            target_latitude = cam.camera_settings.target_latitude
            target_longitude = cam.camera_settings.target_longitude
            target_radius = cam.camera_settings.target_radius
        except Exception:
            pass
        lzresult = lz.calclandingzone()
        if lzresult is None:
            return
        if lzresult.numregions < 5 and lzresult.avgscore < 20000:
            # only accept short lists if they have high scores
            return
        if lzresult.latlon == self.last_lz_latlon:
            # no change
            return
        (lat, lon) = lzresult.latlon
        # check it is within the target radius
        if target_radius > 0:
            dist = cuav_util.gps_distance(lat, lon, target_latitude, target_longitude)
            if dist > target_radius:
                return
            # don't move more than 70m from the center of the search, this keeps us
            # over more of the search area, and further from the fence
            if dist > 70:
                bearing = cuav_util.gps_bearing(lat, lon, target_latitude, target_longitude)
                (lat, lon) = cuav_util.gps_newpos(target_latitude, target_longitude, bearing, 70.0)

        # we may need to fetch the wp list
        if self.last_wp_list is None or self.last_attitude_ms - self.last_wp_list > 120000:
            self.last_wp_list = self.last_attitude_ms
            wpmod.cmd_wp(["list"])
            return
        
        self.last_wp_move = self.last_attitude_ms
        print("Moving search to: ", (lat, lon))
        wpmod.cmd_wp_movemulti([self.cuav_settings.wp_center, self.cuav_settings.wp_start, self.cuav_settings.wp_end], (lat,lon))
        self.wp_move_count += 1
        if self.cuav_settings.wp_land > 0 and self.wp_move_count >= 2 and lzresult.numregions > 10:
            print("Starting landing")
            self.master.waypoint_set_current_send(self.cuav_settings.wp_land)
            self.started_landing = True
            
            
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
        if m.get_type() == 'MISSION_CURRENT':
            self.update_mission()

def init(mpstate):
    '''initialise module'''
    return CUAVCompanionModule(mpstate)
