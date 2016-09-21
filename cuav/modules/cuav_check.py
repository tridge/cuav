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
        super(CUAVModule, self).__init__(mpstate, "CUAV", "CUAV checks")
        self.console.set_status('RPM', 'RPM: --', row=8, fg='black')
        self.console.set_status('RFind', 'RFind: --', row=8, fg='black')
        self.console.set_status('Button', 'Button: --', row=8, fg='black')
        self.console.set_status('ICE', 'ICE: --', row=8, fg='black')
        self.rate_period = mavutil.periodic_event(1.0/15)
        self.button_remaining = None
        self.button_change = None
        self.last_button_update = time.time()
        self.last_target_update = time.time()
        self.button_change_recv_time = 0
        self.button_announce_time = 0
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
              MPSetting('wind_direction', float, 0, 'wind direction (degrees)') ])
        self.add_completion_function('(CUAVCHECKSETTING)', self.cuav_settings.completion)
        self.add_command('cuavcheck', self.cmd_cuavcheck,
                         'cuav check control',
                         ['set (CUAVCHECKSETTING)'])
                         
        #make the initial map menu
        if mp_util.has_wxpython:
            self.menu = MPMenuSubMenu('UAV Challenge', items=[MPMenuCheckbox('Show Landing Zone', 'Show Landing Zone', '# cuavcheck toggleLandingZone'), MPMenuCheckbox('Show Joe Zone', 'Show Joe Zone', '# cuavcheck toggleJoeZone')])
            self.module('map').add_menu(self.menu)
            
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

    def toggle_JoeZone(self):
        '''show/hide the UAV Challenge landing zone around the clicked point'''
        from MAVProxy.modules.mavproxy_map import mp_slipmap
        camera = self.module('camera')
        if camera is None:
            print("camera module is not loaded")
            return
        if camera.camera_settings.target_radius <= 0:
            print("camera module target_radius is not set")
            return
        target = (camera.camera_settings.target_lattitude,
                  camera.camera_settings.target_longitude,
                  camera.camera_settings.target_radius)
        self.target = target
        
        'Create a new layer with given radius around the above point'
        if self.showJoeZone:
            self.mpstate.map.add_object(mp_slipmap.SlipClearLayer('JoeZone'))
            self.mpstate.map.add_object(mp_slipmap.SlipCircle('JoeZoneCircle', layer='JoeZone',
                                                              latlon=(target[0],target[1]), radius=target[2], linewidth=2, color=(0,0,128)))
        else:
            self.mpstate.map.remove_object('JoeZoneCircle')
            self.mpstate.map.remove_object('JoeZone')
                        
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
        else:
            print(usage)
            return            

    def check_parms(self, parms, set=False):
        '''check parameter settings'''
        for p in parms.keys():
            v = self.mav_param.get(p, None)
            if v is None:
                continue
            if abs(v - parms[p]) > 0.0001:
                if set:
                    self.console.writeln('Setting %s to %.1f (currently %.1f)' % (p, parms[p], v), fg='blue')
                    self.master.param_set_send(p, parms[p])
                else:
                    self.console.writeln('%s should be %.1f (currently %.1f)' % (p, parms[p], v), fg='blue')

    def check_rates(self):
        '''check stream rates'''
        parms = {
            "SR0_EXTRA1"    : 1.0,
            "SR0_EXTRA2"    : 2.0,
            "SR0_EXTRA3"    : 1.0,
            "SR0_EXT_STAT"  : 2.0,
            "SR0_PARAMS"    : 10.0,
            "SR0_POSITION"  : 2.0,
            "SR0_RAW_CTRL"  : 1.0,
            "SR0_RAW_SENS"  : 1.0,
            "SR0_RC_CHAN"   : 1.0,
            "SR1_EXTRA1"    : 1.0,
            "SR1_EXTRA2"    : 2.0,
            "SR1_EXTRA3"    : 1.0,
            "SR1_EXT_STAT"  : 2.0,
            "SR1_PARAMS"    : 10.0,
            "SR1_POSITION"  : 2.0,
            "SR1_RAW_CTRL"  : 1.0,
            "SR1_RAW_SENS"  : 1.0,
            "SR1_RC_CHAN"   : 1.0,
            "SR2_EXTRA1"    : 1.0,
            "SR2_EXTRA2"    : 2.0,
            "SR2_EXTRA3"    : 1.0,
            "SR2_EXT_STAT"  : 2.0,
            "SR2_PARAMS"    : 10.0,
            "SR2_POSITION"  : 2.0,
            "SR2_RAW_CTRL"  : 1.0,
            "SR2_RAW_SENS"  : 1.0,
            "SR2_RC_CHAN"   : 1.0,
            "SR3_EXTRA1"    : 1.0,
            "SR3_EXTRA2"    : 2.0,
            "SR3_EXTRA3"    : 1.0,
            "SR3_EXT_STAT"  : 2.0,
            "SR3_PARAMS"    : 10.0,
            "SR3_POSITION"  : 2.0,
            "SR3_RAW_CTRL"  : 1.0,
            "SR3_RAW_SENS"  : 1.0,
            "SR3_RC_CHAN"   : 1.0,
            "FS_GCS_ENABLE" : 0,
            "FS_GCS_ENABL"  : 0,
            }
        self.check_parms(parms, True)

    def idle_task(self):
        '''run periodic tasks'''
        now = time.time()
        if now - self.last_button_update > 0.5:
            self.last_button_update = now
            self.update_button_display()
        if self.last_rpm_update != 0 and now - self.last_rpm_update > 4:
            self.console.set_status('RPM', 'RPM: --', row=8, fg='red')
            self.say("Engine stopped")
            self.last_rpm_update = 0
        if now - self.last_target_update > 1 and self.showJoeZone:
            self.last_target_update = now
            camera = self.module('camera')
            if camera is not None and camera.camera_settings.target_radius > 0:
                target = (camera.camera_settings.target_lattitude,
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

        if m.get_type() == "VFR_HUD":
            flying = False
            if self.status.flightmode == "AUTO" or m.airspeed > 20 or m.groundspeed > 10:
                flying = True
            #if flying and self.settings.mavfwd != 0:
            #    print("Disabling mavfwd for flight")
            #    self.settings.mavfwd = 0

        if m.get_type() == "GLOBAL_POSITION_INT":
            self.update_airspeed_estimate(m)

        if m.get_type() == 'NAMED_VALUE_FLOAT' and m.name == 'BAT3VOLT':
            self.console.set_status('BAT3', 'Bat3: %.2f' % m.value, row=8)
            

        if self.rate_period.trigger():
            self.check_rates()

def init(mpstate):
    '''initialise module'''
    return CUAVModule(mpstate)
