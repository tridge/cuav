#!/usr/bin/env python
'''
CUAV mission control
Andrew Tridgell
'''

from MAVProxy.modules.lib import mp_module

class CUAVModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CUAVModule, self).__init__(mpstate, "CUAV", "CUAV checks")
        self.console.set_status('Bottle', 'Bottle: --', row=8, fg='green')

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

def init(mpstate):
    '''initialise module'''
    return CUAVModule(mpstate)
