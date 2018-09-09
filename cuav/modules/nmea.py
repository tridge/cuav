#!/usr/bin/env python
'''
nmea serial output module
Matthew Ridley
August 2012

UAV outback challenge search and rescue rules
5.15 Situational Awareness Requirement

It is highly desirable that teams provide:
 - an NMEA 0183 serial output with GPRMC and GPGGA sentences for
   aircraft current location

'''

import math
import os
import serial
import socket
import subprocess
import sys
import time

from MAVProxy.modules.lib import mp_module

class NMEAModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(NMEAModule, self).__init__(mpstate,
                                         "NMEA",
                                         "NMEA output",
                                         public=True)
        self.add_command('nmea', self.cmd_nmea, "nmea control")
        self.base_source = NMEASource(mpstate)
        self.secondary_source = None

    def cmd_nmea(self, args):
        '''pass through nmea commands for base source'''
        self.base_source.cmd_nmea(args)

    def mavlink_packet(self, m):
        '''pass through packets for base source'''
        self.base_source.mavlink_packet(m)

    def set_secondary_vehicle_position(self, m):
        '''register secondary vehicle position'''
        if self.secondary_source is None:
            self.add_command('nmea2', self.cmd_nmea, "nmea control")
            self.secondary_source = NMEASource(self.mpstate)

        self.secondary_source.mavlink_packet(m)

class NMEASource():
    '''allows for multiple sources of NMEA information to be received and sent'''

    def __init__(self, mpstate):
        self.port = None
        self.baudrate = 4800
        self.data = 8
        self.parity = 'N'
        self.stop = 1
        self.serial = None
        self.socat = None
        self.log_output = None
        self.udp_output_port = None
        self.udp_output_address = None
        self.output_time = 0.0

        self.num_sat = 0
        self.hdop = 0
        self.altitude = 0
        self.fix_quality = 0
        self.last_time_boot_ms = 0

    def usage(self):
        return """
nmea port [baudrate data parity stop]
e.g.
nmea /dev/ttyS0 115200 8 N 1
nmea socat:GOPEN:/tmp/output
nmea socat:UDP-SENDTO:10.0.1.255:17890
nmea log:/tmp/nmea-log.txt
nmea udp:10.10.10.72:1765
"""

    def cmd_nmea(self, args):
        '''set nmea'''
        if len(args) == 0:
            if self.port is None:
                print("NMEA output port not set")
                print(self.usage())
            else:
                print("NMEA output port %s, %d, %d, %s, %d" % (str(self.port), self.baudrate, self.data, str(self.parity), self.stop))
            return
        if len(args) > 0:
            self.port = str(args[0])
        if len(args) > 1:
            self.baudrate = int(args[1])
        if len(args) > 2:
            self.data = int(args[2])
        if len(args) > 3:
            self.parity = str(args[3])
        if len(args) > 4:
            self.stop = int(args[4])

        if self.serial is not None:
            self.serial.close()
        self.serial = None
        if len(args) > 0:
            if self.port.startswith("/dev/"):
                try:
                    self.serial = serial.Serial(self.port, self.baudrate, self.data, self.parity, self.stop)
                except serial.SerialException as se:
                    print("Failed to open output port %s:%s" % (self.port, se.message))
            elif self.port.startswith("socat:"):
                try:
                    self.start_socat_output(self.port[6:])
                except Exception as se:
                    print("Failed to open socat output %s:%s" % (self.port, se.message))
            elif self.port.startswith("log:"):
                try:
                    self.start_log_output(self.port[4:])
                except Exception as se:
                    print("Failed to open output log %s:%s" % (self.port, se.message))
            elif self.port.startswith("udp:"):
                try:
                    self.start_udp_output(self.port[4:])
                except Exception as se:
                    print("Failed to open udp output %s:%s" % (self.port, se.message))
            else:
                self.serial = open(self.port, mode='w')

    def format_date(self, utc_sec):
        import time
        tm_t = time.gmtime(utc_sec)
        return "%02d%02d%02d" % (tm_t.tm_mday, tm_t.tm_mon, tm_t.tm_year % 100)

    def format_time(self, utc_sec):
        import time
        tm_t = time.gmtime(utc_sec)
        subsecs = utc_sec - int(utc_sec);
        return "%02d%02d%05.3f" % (tm_t.tm_hour, tm_t.tm_min, tm_t.tm_sec + subsecs)

    def format_lat(self, lat):
        deg = abs(lat)
        minutes = (deg - int(deg))*60
        return "%02d%08.5f,%c" % (int(deg), minutes, 'S' if lat < 0 else 'N')

    def format_lon(self, lon):
        deg = abs(lon)
        minutes = (deg - int(deg))*60
        return "%03d%08.5f,%c" % (int(deg), minutes, 'W' if lon < 0 else 'E')

    # tst = "$GPRMC,225446,A,4916.45,N,12311.12,W,000.5,054.7,191194,020.3,E*68"
    # print ("*%02X" % nmea_checksum(tst))

    def nmea_checksum(self, msg):
        d = msg[1:]
        cs = 0
        for i in d:
            cs ^= ord(i)
        return cs

    def format_gga(self, utc_sec, lat, lon, fix, nsat, hdop, alt):
        fmt = "$GPGGA,%s,%s,%s,%01d,%02d,%04.1f,%07.2f,M,0.0,M,,"
        msg = fmt % (self.format_time(utc_sec), self.format_lat(lat), self.format_lon(lon), fix, nsat, hdop, alt)
        return msg + "*%02X" % self.nmea_checksum(msg)

    def format_rmc(self, utc_sec, fix, lat, lon, speed, course):
        fmt = "$GPRMC,%s,%s,%s,%s,%.2f,%.2f,%s,,"
        msg = fmt % (self.format_time(utc_sec), fix, self.format_lat(lat), self.format_lon(lon),
                     speed, course, self.format_date(utc_sec))
        return msg + "*%02X" % self.nmea_checksum(msg)

    def start_socat_output(self, args):
        (self.socat_in, self.socat_out) = os.pipe()
        self.socat = subprocess.Popen(["socat",
                                       "FD:0",
                                       args],
                                      stdin=self.socat_in,
        )

    def start_log_output(self, args):
        filepath = args
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.mpstate.status.logdir, filepath)
        self.log_output = open(filepath, "ab")

    def start_udp_output(self, args):
        (hostname, port) = args.split(":")
        self.udp_output_port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_output_address = (hostname, int(port))

    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        import time

        now_time = time.time()
        if abs(self.output_time - now_time) < 1.0:
            return

        if m.get_type() == 'GPS_RAW_INT':
            self.num_sat = m.satellites_visible
            self.hdop = m.eph/100.0
            self.altitude = m.alt/1000.0
            self.fix_quality = 1 if (m.fix_type > 1) else 0 # 0/1 for (in)valid or 2 DGPS

        if m.get_type() == 'GLOBAL_POSITION_INT':
            if m.time_boot_ms <= self.last_time_boot_ms and self.last_time_boot_ms - m.time_boot_ms < 60000:
                # time going backwards from multiple links
                return
            if m.time_boot_ms - self.last_time_boot_ms < 250:
                # limit to 4Hz
                return
            self.last_time_boot_ms = m.time_boot_ms

            # for GPRMC and GPGGA
            utc_sec = now_time
            fix_status = 'A'
            lat = m.lat/1.0e7
            lon = m.lon/1.0e7

            # for GPRMC
            speed_ms = math.sqrt(m.vx**2+m.vy**2) * 0.01
            knots = speed_ms*1.94384
            course = math.degrees(math.atan2(m.vy,m.vx))
            if course < 0:
                course += 360

            # for GPGGA

            # print format_gga(utc_sec, lat, lon, self.fix_quality, self.num_sat, self.hdop, self.altitude)
            # print format_rmc(utc_sec, fix_status, lat, lon, knots, course)
            gga = self.format_gga(utc_sec, lat, lon, self.fix_quality, self.num_sat, self.hdop, self.altitude)
            rmc = self.format_rmc(utc_sec, fix_status, lat, lon, knots, course)

            self.output_time = now_time
            #print(gga+'\r')
            #print(rmc+'\r')
            #print(self.serial)
            self.output(gga + '\r\n')
            self.output(rmc + '\r\n')

    def output(self, output):
        if self.serial is not None:
            self.serial.write(output)
            self.serial.flush()
        if self.socat is not None:
            os.write(self.socat_out, output)
        if self.log_output is not None:
            timestamped = "%u %s" % (time.time(), output,)
            self.log_output.write(timestamped)
            self.log_output.flush()
        if self.udp_output_port is not None:
            self.udp_output_port.sendto(output, self.udp_output_address)

def init(mpstate):
    '''initialise module'''
    return NMEAModule(mpstate)
