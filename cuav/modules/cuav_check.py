#!/usr/bin/env python
'''
CUAV mission control
Andrew Tridgell
'''

from MAVProxy.modules.lib import mp_module
from pymavlink import mavutil

class CUAVModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CUAVModule, self).__init__(mpstate, "CUAV", "CUAV checks")
        self.console.set_status('Bottle', 'Bottle: --', row=8, fg='green')
        self.rate_period = mavutil.periodic_event(1.0/15)

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
            "SR1_EXTRA1"    : 3.0,
            "SR1_EXTRA2"    : 2.0,
            "SR1_EXTRA3"    : 2.0,
            "SR1_EXT_STAT"  : 2.0,
            "SR1_PARAMS"    : 10.0,
            "SR1_POSITION"  : 4.0,
            "SR1_RAW_CTRL"  : 2.0,
            "SR1_RAW_SENS"  : 1.0,
            "SR1_RC_CHAN"   : 1.0,
            "SR2_EXTRA1"    : 4.0,
            "SR2_EXTRA2"    : 4.0,
            "SR2_EXTRA3"    : 4.0,
            "SR2_EXT_STAT"  : 4.0,
            "SR2_PARAMS"    : 10.0,
            "SR2_POSITION"  : 4.0,
            "SR2_RAW_CTRL"  : 4.0,
            "SR2_RAW_SENS"  : 4.0,
            "SR2_RC_CHAN"   : 4.0
            }
        self.check_parms(parms, True)


    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        if m.get_type() == "SERVO_OUTPUT_RAW":
            bottle = m.servo8_raw
            if bottle == 950:
                self.console.set_status('Bottle', 'Bottle: HELD', row=8, fg='green')
            elif bottle == 1500:
                self.console.set_status('Bottle', 'Bottle: DROP', row=8, fg='red')
            else:
                self.console.set_status('Bottle', 'Bottle: %u' % bottle, row=8, fg='red')

        if m.get_type() == "VFR_HUD":
            flying = False
            if self.status.flightmode == "AUTO" or m.airspeed > 20 or m.groundspeed > 10:
                flying = True
            if flying and self.settings.mavfwd != 0:
                print("Disabling mavfwd for flight")
                self.settings.mavfwd = 0

        if self.rate_period.trigger():
            self.check_rates()

def init(mpstate):
    '''initialise module'''
    return CUAVModule(mpstate)
