#!/usr/bin/env python
'''
CUAV mission control
Andrew Tridgell
'''

from MAVProxy.modules.lib import mp_module
from pymavlink import mavutil
import time, math

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
        self.button_change_recv_time = 0
        self.button_announce_time = 0
        self.last_rpm_update = 0
        self.last_rpm_value = None

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
            "SR1_EXTRA1"    : 2.0,
            "SR1_EXTRA2"    : 2.0,
            "SR1_EXTRA3"    : 2.0,
            "SR1_EXT_STAT"  : 2.0,
            "SR1_PARAMS"    : 10.0,
            "SR1_POSITION"  : 2.0,
            "SR1_RAW_CTRL"  : 2.0,
            "SR1_RAW_SENS"  : 1.0,
            "SR1_RC_CHAN"   : 1.0,
            "SR2_EXTRA1"    : 2.0,
            "SR2_EXTRA2"    : 2.0,
            "SR2_EXTRA3"    : 2.0,
            "SR2_EXT_STAT"  : 2.0,
            "SR2_PARAMS"    : 10.0,
            "SR2_POSITION"  : 2.0,
            "SR2_RAW_CTRL"  : 1.0,
            "SR2_RAW_SENS"  : 1.0,
            "SR2_RC_CHAN"   : 1.0,
            "SR3_EXTRA1"    : 2.0,
            "SR3_EXTRA2"    : 2.0,
            "SR3_EXTRA3"    : 2.0,
            "SR3_EXT_STAT"  : 2.0,
            "SR3_PARAMS"    : 10.0,
            "SR3_POSITION"  : 2.0,
            "SR3_RAW_CTRL"  : 2.0,
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
            self.last_rpm_value = None

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

        if m.get_type() == "RC_CHANNELS":
            v = m.chan7_raw
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

        if self.rate_period.trigger():
            self.check_rates()

def init(mpstate):
    '''initialise module'''
    return CUAVModule(mpstate)
