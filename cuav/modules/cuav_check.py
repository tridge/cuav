#!/usr/bin/env python
'''
CUAV mission control
Andrew Tridgell
'''

from MAVProxy.modules.lib import mp_module
from pymavlink import mavutil
import time, math, functools
from MAVProxy.modules.lib import mp_settings
from MAVProxy.modules.lib import mp_util

if mp_util.has_wxpython:
    from MAVProxy.modules.lib.mp_menu import *

class CUAVModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CUAVModule, self).__init__(mpstate, "CUAV", "CUAV checks", public=True)
        self.console.set_status('RPM', 'RPM: --', row=8, fg='black')
        self.console.set_status('RFind', 'RFind: --', row=8, fg='black')
        self.console.set_status('Button', 'Button: --', row=8, fg='black')
        self.console.set_status('ICE', 'ICE: --', row=8, fg='black')
        self.console.set_status('FuelPump', 'FuelPump: --', row=8, fg='black')
        self.console.set_status('DNFZ', 'DNFZ -- --', row=6, fg='black')
        self.rate_period = mavutil.periodic_event(1.0/15)
        self.button_remaining = None
        self.button_change = None
        self.last_button_update = time.time()
        self.last_target_update = time.time()
        self.button_change_recv_time = 0
        self.button_announce_time = 0

        self.fuel_change = None
        self.last_fuel_update = time.time()
        self.fuel_change_recv_time = 0
        
        self.last_rpm_update = 0
        self.last_rpm_value = None
        self.last_rpm_announce = 0
        self.showLandingZone = 0
        self.showJoeZone = True
        self.target = None

        from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
        self.cuav_settings = MPSettings(
            [ MPSetting('rpm_threshold', int, 6000, 'RPM Threshold'),
              MPSetting('wind_speed', float, 0, 'wind speed (m/s)'),
              MPSetting('wind_direction', float, 0, 'wind direction (degrees)'),
              MPSetting('button_pin', float, 0, 'button pin'),
              MPSetting('fuel_pin', float, 1, 'fuel pin'),
              MPSetting('wp_center', int, 2, 'center search USER number'),
              MPSetting('wp_start', int, 1, 'start search USER number'),
              MPSetting('wp_end', int, 3, 'end search USER number'),
              MPSetting('wp_land',int, 4, 'landing start USER number') ])
        self.add_completion_function('(CUAVCHECKSETTING)', self.cuav_settings.completion)
        self.add_command('cuavcheck', self.cmd_cuavcheck,
                         'cuav check control',
                         ['checkparams',
                          'movetarget',
                          'set (CUAVCHECKSETTING)'])

        #make the initial map menu
        if mp_util.has_wxpython and self.module('map'):
            self.menu = MPMenuSubMenu('UAV Challenge', items=[MPMenuCheckbox('Show Landing Zone', 'Show Landing Zone', '# cuavcheck toggleLandingZone'), MPMenuCheckbox('Show Joe Zone', 'Show Joe Zone', '# cuavcheck toggleJoeZone')])
            self.module('map').add_menu(self.menu)

    def find_user_wp(self, wploader, n):
        '''find a USER_ waypoint number'''
        for i in range(1, wploader.count()):
            wp = wploader.wp(i)
            if wp.command == mavutil.mavlink.MAV_CMD_USER_1 + (n-1):
                # the USER_n waypoint is just before the waypoint to use
                return i+1
        return None
            
    def toggle_LandingZone(self):
        '''show/hide the UAV Challenge landing zone around the clicked point'''
        from MAVProxy.modules.mavproxy_map import mp_slipmap
        pos = self.module('map').click_position
        'Create a new layer of two circles - at 30m and 80m radius around the above point'
        if(self.showLandingZone):
            self.mpstate.map.add_object(mp_slipmap.SlipClearLayer('LandingZone'))
            self.mpstate.map.add_object(mp_slipmap.SlipCircle('LandingZoneInner', layer='LandingZone', latlon=pos, radius=30, linewidth=2, color=(0,0,255)))
            self.mpstate.map.add_object(mp_slipmap.SlipCircle('LandingZoneOuter', layer='LandingZone', latlon=pos, radius=80, linewidth=2, color=(0,0,255)))
        else:
            self.mpstate.map.remove_object('LandingZoneInner')
            self.mpstate.map.remove_object('LandingZoneOuter')
            self.mpstate.map.remove_object('LandingZone')

    def show_JoeZone(self):
        '''show the UAV Challenge landing zone around the clicked point'''
        from MAVProxy.modules.mavproxy_map import mp_slipmap
        camera = self.module('camera_ground')
        if camera is None:
            print("camera_ground module is not loaded")
            return
        target = (camera.camera_settings.target_latitude,
                  camera.camera_settings.target_longitude,
                  camera.camera_settings.target_radius)
        self.target = target

        for m in self.module_matching('map*'):
            m.map.add_object(mp_slipmap.SlipClearLayer('JoeZone'))
            m.map.add_object(mp_slipmap.SlipCircle('JoeZoneCircle', layer='JoeZone',
                                                   latlon=(target[0],target[1]), radius=target[2], linewidth=2, color=(0,0,128)))

    def hide_JoeZone(self):
        '''hide the UAV Challenge landing zone around the clicked point'''
        from MAVProxy.modules.mavproxy_map import mp_slipmap
        for m in self.module_matching('map*'):
            m.map.remove_object('JoeZoneCircle')
            m.map.remove_object('JoeZone')
            
    def toggle_JoeZone(self):
        '''show/hide the UAV Challenge landing zone around the clicked point'''
        from MAVProxy.modules.mavproxy_map import mp_slipmap
        camera = self.module('camera_ground')
        if self.mpstate.map is None:
            print("Map module not loaded")
            return
        if camera is None:
            print("camera_ground module is not loaded")
            return
        if camera.camera_settings.target_radius <= 0:
            print("camera_ground module target_radius is not set")
            return
        target = (camera.camera_settings.target_latitude,
                  camera.camera_settings.target_longitude,
                  camera.camera_settings.target_radius)
        self.target = target

        if self.showJoeZone:
            self.show_JoeZone()
        else:
            self.hide_JoeZone()
            
    def cmd_cuavcheck(self, args):
        '''handle cuavcheck commands'''
        usage = 'Usage: cuavcheck <set>'
        if len(args) == 0:
            print(usage)
            return
        if args[0] == "set":
            self.cuav_settings.command(args[1:])
        elif args[0] == "toggleLandingZone":
            self.showLandingZone = not self.showLandingZone
            self.toggle_LandingZone()
        elif args[0] == "toggleJoeZone":
            self.showJoeZone = not self.showJoeZone
            self.toggle_JoeZone()
        elif args[0] == "checkparams":
            if self.check_parameters():
                print("Parameters OK")
            else:
                print("Parameters bad")
        elif args[0] == "movetarget":
            self.move_target()
        else:
            print(usage)
            return

    def move_target(self):
        '''move target waypoints'''
        wpmod = self.module('wp')
        wploader = wpmod.wploader

        wp_start = self.find_user_wp(wploader, 1)
        wp_center = self.find_user_wp(wploader, 2)
        wp_end = self.find_user_wp(wploader, 3)
        if (wp_center is None or
            wp_start is None or
            wp_end is None):
            print("Target WPs not in mission")
            return
        latlon = self.module('map').click_position
        if latlon is None:
            print("No click position")
            return
        print("Moving %u waypoints" % (wp_end + 1 - wp_start), wp_center, wp_start, wp_end)
        wpmod.cmd_wp_movemulti([wp_center, wp_start, wp_end], latlon)

    def check_parms(self, parms, set=False):
        '''check parameter settings'''
        ret = True
        for p in parms.keys():
            v = self.mav_param.get(p, None)
            if v is None:
                self.console.writeln("Parameter %s unavailable" % p)
                continue
            if abs(v - parms[p]) > 0.0001:
                if set:
                    self.console.writeln('Setting %s to %.1f (currently %.1f)' % (p, parms[p], v), fg='blue')
                    self.master.param_set_send(p, parms[p])
                else:
                    self.console.writeln('%s should be %.1f (currently %.1f)' % (p, parms[p], v), fg='blue')
                ret = False
        return ret

    def check_parameters(self):
        '''check key parameters'''
        # first see if this is a quadplane
        v = self.mav_param.get('Q_ENABLE',None)
        if v is None:
            self.console.writeln('Q_ENABLE set available')
            return False
        if int(v) == 0:
            # this is the relay aircraft
            return self.check_parameters_relay()
        else:
            return self.check_parameters_retrieval()

    def check_parameters_relay(self):
        # relay aircraft should have low rates on SR1
        rates = {
            "SR1_EXTRA1"    : 1.0,
            "SR1_EXTRA2"    : 1.0,
            "SR1_EXTRA3"    : 1.0,
            "SR1_EXT_STAT"  : 2.0,
            "SR1_POSITION"  : 2.0,
            "SR1_RAW_CTRL"  : 1.0,
            "SR1_RAW_SENS"  : 1.0,
            "SR1_RC_CHAN"   : 1.0
            }
        ret = self.check_parms(rates, True)
        # some other key parameters, not auto-set
        keyparams = {
            "FS_GCS_ENABL"  : 0,
            "AVD_W_ACTION"  : 2,
            "FENCE_AUTOENABLE" : 1,
            "RC_OPTIONS" : 4,
            "SERIAL1_PROTOCOL" : 2,
            }
        if not self.check_parms(keyparams, False):
            ret = False
        return ret
            
    def check_parameters_retrieval(self):
        # retrieval aircraft should have low rates on SR1, higher rates on SR2
        rates = {
            "SR1_EXTRA1"    : 1.0,
            "SR1_EXTRA2"    : 1.0,
            "SR1_EXTRA3"    : 1.0,
            "SR1_EXT_STAT"  : 2.0,
            "SR1_POSITION"  : 2.0,
            "SR1_RAW_CTRL"  : 1.0,
            "SR1_RAW_SENS"  : 1.0,
            "SR1_RC_CHAN"   : 1.0,
            "SR2_EXTRA1"    : 6.0,
            "SR2_EXTRA2"    : 4.0,
            "SR2_EXTRA3"    : 4.0,
            "SR2_EXT_STAT"  : 4.0,
            "SR2_POSITION"  : 6.0,
            "SR2_RAW_CTRL"  : 4.0,
            "SR2_RAW_SENS"  : 4.0,
            "SR2_RC_CHAN"   : 1.0,
            }
        ret = self.check_parms(rates, True)
        # some other key parameters, not auto-set
        keyparams = {
            "Q_OPTIONS" : 8,
            "AVD_ENABLE" : 1,
            "ADSB_ENABLE" : 1,
            "FS_GCS_ENABL"  : 0,
            "AVD_W_ACTION"  : 2,
            "FENCE_AUTOENABLE" : 1,
            "RC_OPTIONS" : 4,
            "SERIAL1_PROTOCOL" : 2,
            "SERIAL2_PROTOCOL" : 2,
            }
        if not self.check_parms(keyparams, False):
            ret = False
        return ret
            
    def idle_task(self):
        '''run periodic tasks'''
        now = time.time()
        if now - self.last_button_update > 0.5:
            self.last_button_update = now
            self.update_button_display()
        if now - self.last_fuel_update > 1.0:
            self.last_fuel_update = now
            self.update_fuel_display()
        if self.last_rpm_update != 0 and now - self.last_rpm_update > 4:
            self.console.set_status('RPM', 'RPM: --', row=8, fg='red')
            self.say("Engine stopped")
            self.last_rpm_update = 0
        if now - self.last_target_update > 1 and self.showJoeZone:
            self.last_target_update = now
            camera = self.module('camera_ground')
            if camera is not None and camera.camera_settings.target_radius > 0:
                target = (camera.camera_settings.target_latitude,
                          camera.camera_settings.target_longitude,
                          camera.camera_settings.target_radius)
                if target != self.target:
                    self.showJoeZone = False
                    self.toggle_JoeZone()

    def update_button_display(self):
        '''update the Button display on console'''
        if self.button_change is None:
            return
        now = time.time()
        time_since_change = (self.button_change.time_boot_ms - self.button_change.last_change_ms) * 0.001
        time_since_change += now - self.button_change_recv_time
        if time_since_change > 60:
            color = 'black'
            self.button_remaining = 0
        else:
            color = 'red'
            self.button_remaining = 60 - time_since_change
        remaining = int(self.button_remaining)
        self.console.set_status('Button', 'Button: %u' % remaining, row=8, fg=color)
        if remaining > 0 and now - self.button_announce_time > 60:
                self.say("Button pressed")
                self.button_announce_time = now
                return
        if now - self.button_announce_time >= 10 and remaining % 10 == 0 and time_since_change < 65:
            self.say("%u seconds" % remaining)
            self.button_announce_time = now

    def update_fuel_display(self):
        '''update the fuel display on console'''
        if self.fuel_change is None:
            return
        now = time.time()
        time_since_change = (self.fuel_change.time_boot_ms - self.fuel_change.last_change_ms) * 0.001
        time_since_change += now - self.fuel_change_recv_time
        self.console.set_status('FuelPump', 'FuelPump: %u' % int(time_since_change), row=8, fg='black')
            
    def rpm_check(self, m):
        '''check for correct RPM range'''
        thr = self.master.field('VFR_HUD', 'throttle', 0)
        if thr >= 100 and m.rpm1 < self.cuav_settings.rpm_threshold:
            self.console.set_status('RPM', 'RPM: %u' % m.rpm1, row=8, fg='red')
            now = time.time()
            if now - self.last_rpm_announce > 20:
                self.say("RPM warning")
                self.last_rpm_announce = now

    def update_airspeed_estimate(self, m):
        '''update airspeed estimate for helicopters'''
        if self.cuav_settings.wind_speed <= 0:
            return
        from pymavlink.rotmat import Vector3
        wind = Vector3(self.cuav_settings.wind_speed*math.cos(math.radians(self.cuav_settings.wind_direction)),
                       self.cuav_settings.wind_speed*math.sin(math.radians(self.cuav_settings.wind_direction)), 0)
        ground = Vector3(m.vx*0.01, m.vy*0.01, 0)
        airspeed = ground + wind
        self.console.set_status('AirspeedEstimate', 'AirspeedEstimate: %u m/s' % airspeed.length(), row=8)


    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        now = time.time()
        if m.get_type() == "BUTTON_CHANGE":
            if m.state & (1 << self.cuav_settings.fuel_pin):
                self.fuel_change = m
                self.fuel_change_recv_time = now
                self.update_fuel_display()
            if m.state & (1 << self.cuav_settings.button_pin):
                if self.button_change is None or m.last_change_ms != self.button_change.last_change_ms:
                    print("button change", m.state)
                if self.button_change is not None:
                    if (m.time_boot_ms < self.button_change.time_boot_ms and
                        self.button_change.time_boot_ms - m.time_boot_ms < 30000):
                        # discard repeated packet from another link if older by less than 30s
                        return
                self.button_change = m
                self.button_change_recv_time = now
                self.update_button_display()

        if m.get_type() == "RPM":
            self.console.set_status('RPM', 'RPM: %u' % m.rpm1, row=8)
            self.last_rpm_update = now
            if m.rpm1 > 50:
                if self.last_rpm_value is None:
                    self.say("Engine started")
                self.last_rpm_value = m.rpm1
                self.rpm_check(m)

        if m.get_type() == "RC_CHANNELS":
            v = self.mav_param.get('ICE_START_CHAN', None)
            if v is None:
                return
            v = getattr(m, 'chan%u_raw' % v)
            if v <= 1300:
                self.console.set_status('ICE', 'ICE: OFF', row=8, fg='red')
            elif v >= 1700:
                self.console.set_status('ICE', 'ICE: ON', row=8, fg='blue')
            else:
                self.console.set_status('ICE', 'ICE: AUTO', row=8, fg='green')

        if m.get_type() == "RANGEFINDER" and 'ATTITUDE' in self.master.messages:
            a = self.master.messages['ATTITUDE']
            dist = m.distance * math.cos(a.roll) * math.cos(a.pitch)
            self.console.set_status('RFind', 'RFind: %.1fm %uft' % (dist, dist*3.28084), row=8)

        if m.get_type() == "GLOBAL_POSITION_INT":
            self.update_airspeed_estimate(m)

        if m.get_type() == 'NAMED_VALUE_FLOAT' and m.name == 'BAT3VOLT':
            self.console.set_status('BAT3', 'Bat3: %.2f' % m.value, row=8)

        if m.get_type() == 'COLLISION':
            if m.action == 0:
                color = 'green'
            elif m.action == 1:
                color = 'blue'
            elif m.action == 2:
                color = 'orange'
            elif m.action == 3:
                color = 'darkorange'
            elif m.action == 4:
                color = 'darkred'
            else:
                color = 'red'
            self.console.set_status('DNFZ', 'DNFZ %d %.0fm %.0fm %u' % (
                m.id, m.horizontal_minimum_delta, m.altitude_minimum_delta, m.src), row=6, fg=color)

        if self.rate_period.trigger():
            self.check_parameters()

def init(mpstate):
    '''initialise module'''
    return CUAVModule(mpstate)
