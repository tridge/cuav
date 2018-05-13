"""
  MAVProxy checklists

  uses lib/libchecklist.py for UI
"""

import os, sys, math

mpstate = None

from MAVProxy.modules.lib import mp_module
import libchecklist

class ChecklistModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(ChecklistModule, self).__init__(mpstate, "checklist", "checklist handling")
        self.checklist = libchecklist.UI()

    def mavlink_packet(self, msg):
        '''handle an incoming mavlink packet'''
        if not isinstance(self.checklist, libchecklist.UI):
            return
        if not self.checklist.is_alive():
            self.libchecklist = None
            return

        type = msg.get_type()
        master = self.master

        '''beforeEngineList - Altitude lock'''
        if type == 'VFR_HUD':
            if msg.alt > 0 and msg.alt < 5000:
                self.checklist.set_status("Altitude lock", 1)
            else:
                self.checklist.set_status("Altitude lock", 0)

        if type == 'ATTITUDE':
            '''beforeEngineList - UAV Level'''
            if (math.fabs(math.degrees(msg.pitch)) < 3 and math.fabs(math.degrees(msg.roll)) < 3):
                self.checklist.set_status("UAV Level", 1)
            else:
                self.checklist.set_status("UAV Level", 0)

        if type == 'SYS_STATUS':
            '''beforeEngineList - Avionics Battery'''
            if msg.battery_remaining > 70:
                self.checklist.set_status("Avionics Battery", 1)
            else:
                self.checklist.set_status("Avionics Battery", 0)

        '''beforeEngineList - Waypoints Loaded'''
        if type == 'HEARTBEAT':
            if self.module('wp').wploader.count() == 0:
                self.checklist.set_status("Waypoints Loaded", 0)
            else:
                self.checklist.set_status("Waypoints Loaded", 1)

        '''beforeEngineList - Trim set from controller'''
        if type == 'HEARTBEAT':
            if 'RC1_TRIM' in self.mav_param and int(self.mav_param['RC1_TRIM']) == 0:
                self.checklist.set_status("Trim set from controller", 0)
            elif 'RC2_TRIM' in self.mav_param and int(self.mav_param['RC2_TRIM']) == 0:
                self.checklist.set_status("Trim set from controller", 0)
            elif 'RC3_TRIM' in self.mav_param and int(self.mav_param['RC3_TRIM']) == 0:
                self.checklist.set_status("Trim set from controller", 0)
            elif 'RC4_TRIM' in self.mav_param and int(self.mav_param['RC4_TRIM']) == 0:
                self.checklist.set_status("Trim set from controller", 0)
            else:
                self.checklist.set_status("Trim set from controller", 1)

        '''beforeTakeoffList - Compass active'''
        if type == 'GPS_RAW_INT':
            if (math.fabs(msg.cog - master.field('VFR_HUD', 'heading', '-')) < 10 or
                math.fabs(msg.cog - master.field('VFR_HUD', 'heading', '-')) > 355):
                self.checklist.set_status("Compass active", 1)
            else:
                self.checklist.set_status("Compass active", 0)

        '''beforeCruiseList - Airspeed > 10 m/s , Altitude > 30 m'''
        if type == 'VFR_HUD':
            if self.status.altitude > 30:
                self.checklist.set_status("Altitude > 30 m", 1)
            else:
                self.checklist.set_status("Altitude > 30 m", 0)
            if msg.airspeed > 10 or msg.groundspeed > 10:
                self.checklist.set_status("Airspeed > 10 m/s", 1)
            else:
                self.checklist.set_status("Airspeed > 10 m/s", 0)

    def unload(self):
        if self.checklist:
            self.checklist.close()

def init(mpstate):
    '''initialise module'''
    return ChecklistModule(mpstate)
