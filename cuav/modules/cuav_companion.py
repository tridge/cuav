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
        self.last_wp_move_ms = 0
        self.add_command('cuavled', self.cmd_cuavled, "cuav led command", ['<red|green|flash|off|refresh>'])
        from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
        self.cuav_settings = MPSettings(
            [ MPSetting('wp_center', int, 2, 'center search USER number'),
              MPSetting('wp_start', int, 1, 'start search USER number'),
              MPSetting('wp_end', int, 3, 'end search USER number'),
              MPSetting('wp_land',int, 4, 'landing start USER number'),
              MPSetting('lookahead_default',int, 500, 'avoidance lookahead main'),
              MPSetting('lookahead_search',int, 300, 'avoidance lookahead search'),
              MPSetting('margin_exc_default',int, 100, 'avoidance exc main'),
              MPSetting('margin_exc_search',int, 35, 'avoidance exc margin'),
              MPSetting('auto_mission',bool, True, 'enable auto mission code') ])
        self.add_command('cuav', self.cmd_cuav,
                         'cuav companion control',
                         ['set (CUAVSETTING)'])
        self.add_completion_function('(CUAVSETTING)', self.cuav_settings.completion)
        self.wp_move_count = 0
        self.last_lz_latlon = None
        self.last_wp_list_ms = 0
        self.started_landing = False
        self.updated_waypoints = False
        self.last_buzzer = time.time()

    def send_message(self, msg):
        '''send a msg to GCS for display'''
        msg = "cuav: " + msg
        print(msg)
        try:
            cam = self.module('camera_air')
            cam.send_message(msg)
        except Exception as ex:
            print("err: ", ex)
        
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
        #self.set_relay(0, state[0])
        #self.set_relay(1, state[1])

        pattern = [0] * 24
        plen = 3
        if state[2] == 'RED':
            pattern[0] = 255
        elif state[2] == 'GREEN':
            pattern[1] = 255
        elif state[2] == 'FLASH':
            pattern[0] = 255
            pattern[1] = 255
            pattern[3] = 2 # 2Hz flash
            plen = 4
        self.master.mav.led_control_send(self.settings.target_system,
                                         self.settings.target_component,
                                         0, 0, plen, pattern)
            
        if state == LED_FLASH:
            # also play warning tune
            self.master.mav.play_tune_send(self.settings.target_system,
                                           self.settings.target_component,
                                           'AAAAAA', '')
            self.last_buzzer = time.time()

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
            try:
                wpmod = self.module('wp')
                wploader = wpmod.wploader
                wpcur = self.master.messages['MISSION_CURRENT'].seq
                wp = wploader.wp(wpcur)
                if wp.command == mavutil.mavlink.MAV_CMD_NAV_DELAY_AIRSPACE_CLEAR:
                    led_state = LED_FLASH
            except Exception as ex:
                pass
        if led_state != self.led_state:
            self.set_leds(led_state)
            try:
                self.send_message("Changing LEDs to: %s" % led_state[2])
            except Exception as ex:
                pass
        now = time.time()
        if led_state[2] == 'FLASH' and now - self.last_buzzer > 5:
            self.last_buzzer = now
            self.master.mav.play_tune_send(self.settings.target_system,
                                           self.settings.target_component,
                                           'AAAAAA','')


    def find_user_wp(self, wploader, n):
        '''find a USER_ waypoint number'''
        for i in range(1, wploader.count()):
            wp = wploader.wp(i)
            if wp.command == mavutil.mavlink.MAV_CMD_USER_1 and wp.param1 == n:
                # the USER_1 waypoint is just before the waypoint to use
                return i+1
        return None

    def update_mission(self, m):
        '''update mission status'''
        if not self.cuav_settings.auto_mission:
            return

        wpmod = self.module('wp')
        wploader = wpmod.wploader
        if wploader.count() < 2 and self.last_attitude_ms - self.last_wp_list_ms > 5000:
            self.last_wp_list_ms = self.last_attitude_ms
            wpmod.cmd_wp(["list"])

        wp_start = self.find_user_wp(wploader, self.cuav_settings.wp_start)
        wp_center = self.find_user_wp(wploader, self.cuav_settings.wp_center)
        wp_end = self.find_user_wp(wploader, self.cuav_settings.wp_end)

        if (wp_center is None or
            wp_start is None or
            wp_end is None):
            # not configured
            return

        if m.seq >= wp_start and m.seq <= wp_end:
            lookahead = self.cuav_settings.lookahead_search
            margin_exc = self.cuav_settings.margin_exc_search
        else:
            lookahead = self.cuav_settings.lookahead_default
            margin_exc = self.cuav_settings.margin_exc_default

        v = self.mav_param.get('AVD_LOOKAHEAD', None)
        if v is not None and abs(v - lookahead) > 1:
            self.send_message("Set AVD_LOOKAHEAD %u" % lookahead)
            self.master.param_set_send('AVD_LOOKAHEAD', lookahead)

        v = self.mav_param.get('AVD_MARGIN_EXCL', None)
        if v is not None and abs(v - margin_exc) > 1:
            self.send_message("Set AVD_MARGIN_EXCL %u" % margin_exc)
            self.master.param_set_send('AVD_MARGIN_EXCL', margin_exc)
            
        # run every 5 seconds
        if self.last_attitude_ms - self.last_mission_check_ms < 5000:
            return
        self.last_mission_check_ms = self.last_attitude_ms

        if self.updated_waypoints:
            cam = self.module('camera_air')
            if wpmod.loading_waypoints:
                self.send_message("listing waypoints")                
                wpmod.cmd_wp(["list"])
            else:
                self.send_message("sending waypoints")
                self.updated_waypoints = False
                wploader.save("newwp.txt")
                cam.send_file("newwp.txt")
        
        if self.started_landing:
            # no more to do
            return

        if self.last_attitude_ms - self.last_wp_move_ms < 2*60*1000:
            # only move waypoints every 2 minutes
            return

        try:
            cam = self.module('camera_air')
            lz = cam.lz
            target_latitude = cam.camera_settings.target_latitude
            target_longitude = cam.camera_settings.target_longitude
            target_radius = cam.camera_settings.target_radius
        except Exception:
            self.send_message("target not set")
            return
        
        lzresult = lz.calclandingzone()
        if lzresult is None:
            return
        
        self.send_message("lzresult nr:%u avgscore:%u" % (lzresult.numregions, lzresult.avgscore))
        
        if lzresult.numregions < 5 and lzresult.avgscore < 20000:
            # only accept short lists if they have high scores
            return
        
        (lat, lon) = lzresult.latlon
        # check it is within the target radius
        if target_radius > 0:
            dist = cuav_util.gps_distance(lat, lon, target_latitude, target_longitude)
            self.send_message("dist %u" % dist)
            if dist > target_radius:
                return
            # don't move more than 70m from the center of the search, this keeps us
            # over more of the search area, and further from the fence
            max_move = target_radius
            if self.wp_move_count == 0:
                # don't move more than 50m from center on first move
                max_move = 35
            if self.wp_move_count == 1:
                # don't move more than 80m from center on 2nd move
                max_move = 80
            if dist > max_move:
                bearing = cuav_util.gps_bearing(target_latitude, target_longitude, lat, lon)
                (lat, lon) = cuav_util.gps_newpos(target_latitude, target_longitude, bearing, max_move)

        # we may need to fetch the wp list
        if self.last_attitude_ms - self.last_wp_list_ms > 120000 or wpmod.loading_waypoints:
            self.last_wp_list_ms = self.last_attitude_ms
            self.send_message("fetching waypoints")
            wpmod.cmd_wp(["list"])
            return
        
        self.last_wp_move_ms = self.last_attitude_ms
        self.wp_move_count += 1
        self.send_message("Moving search to: (%f,%f) %u" % (lat, lon, self.wp_move_count))
        wpmod.cmd_wp_movemulti([wp_center, wp_start, wp_end], (lat,lon))

        wp_land = self.find_user_wp(wploader, self.cuav_settings.wp_land)
        if (wp_land is not None and
            self.wp_move_count >= 3 and
            lzresult.numregions > 10 and
            self.master.flightmode == "AUTO"):
            self.send_message("Starting landing")
            self.master.waypoint_set_current_send(wp_land)
            self.started_landing = True
        self.updated_waypoints = True
            
            
    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        now = time.time()
        if m.get_type() == "BUTTON_CHANGE":
            if m.state & 1:
                time_since = max(m.time_boot_ms - m.last_change_ms,0) * 1.0e-3
                self.button_change_time = time.time() - time_since
                self.update_led_state()
        if m.get_type() == 'HEARTBEAT':
            self.update_led_state()
        if m.get_type() == 'COMMAND_ACK' and m.command == mavutil.mavlink.MAV_CMD_DO_SET_RELAY and self.ack_wait > 0:
            self.ack_wait -= 1
            if self.ack_wait == 0:
                self.send_message("LEDs updated: %s" % self.led_state[2])
        if m.get_type() == 'ATTITUDE':
            if m.time_boot_ms < self.last_attitude_ms:
                self.send_message("time wrapped")
                self.led_state = None
                self.last_mission_check_ms = 0
                self.last_wp_move_ms = 0
                self.last_wp_list_ms = 0
                self.button_change_time = 0
            self.last_attitude_ms = m.time_boot_ms
        if m.get_type() == 'MISSION_CURRENT':
            self.update_mission(m)

def init(mpstate):
    '''initialise module'''
    return CUAVCompanionModule(mpstate)
