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
from MAVProxy.modules.lib import mp_settings

class NMEAModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(NMEAModule, self).__init__(mpstate,
                                         "NMEA",
                                         "NMEA output",
                                         public=True,
                                         multi_instance=True)
        cmdname = "nmea"
        if self.instance > 1:
            cmdname += "%u" % self.instance
        self.add_command(cmdname,
                         self.cmd_nmea,
                         "nmea control",
                         ['<status|set>'])

        self.serial_line = None
        self.serial = None
        self.socat = None
        self.log_output = None
        self.log_output_filepath = None
        self.udp_output_port = None
        self.udp_output_address = None
        self.output_time = 0.0

        self.num_sat = 21
        self.hdop = 1.21
        self.fix_quality = 1
        self.last_time_boot_ms = 0

        self.nmea_settings = mp_settings.MPSettings(
            [('target_system', int, 1),
             ('wgs84_to_amsl', float, -41.2),
            ]
        )
        self.add_completion_function('(NMEASETTING)',
                                     self.nmea_settings.completion)

        self.sent_count = 0

    def usage(self):
        return """
nmea serial /dev/ttyS0 115200 8 N 1
nmea udp 10.10.10.72:1765
nmea log /tmp/nmea-log.txt
nmea log log.txt
nmea socat GOPEN:/tmp/output
nmea socat UDP-SENDTO:10.0.1.255:17890
"""

    def cmd_status(self, rest):
        print("NMEA-serial: %s" % str(self.serial_line))
        print("NMEA-socat: %s" % str(self.socat))
        print("NMEA-log-output: %s" % str(self.log_output_filepath))
        print("NMEA-udp-output: %s" % (str(self.udp_output_address)))

    def cmd_nmea(self, args):
        '''set nmea'''
        if len(args) == 0:
            self.cmd_status(args)
            return
        cmd = args[0]
        args = args[1:]
        if cmd == 'status':
            self.cmd_status(args)
        elif cmd == 'set':
            self.nmea_settings.command(args)
        elif cmd == 'udp':
            self.cmd_udp(args)
        elif cmd == 'log':
            self.cmd_log(args)
        elif cmd == 'socat':
            self.cmd_log(args)
        elif cmd == 'serial':
            self.cmd_serial(args)
        else:
            print("Unknown command (%s)" % cmd)
        return

    def cmd_serial(self, args):
        if len(args) != 0 and len(args) < 5:
            print(self.usage())
            return

        if self.serial is not None:
            self.serial.close()
        self.serial = None

        if len(args) == 0:
            return

        port = str(args[0])
        baudrate = int(args[1])
        data = int(args[2])
        parity = str(args[3])
        stop = int(args[4])
        self.serial_line = ":".join(args)

        try:
            self.serial = serial.Serial(port, baudrate, data, parity, stop)
        except serial.SerialException as se:
            print("Failed to open serial %s:%s" % (self.serial_line, se.message))

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
    # From: http://aprs.gids.nl/nmea/#gga
    # eg3. $GPGGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx*hh
    # 1    = UTC of Position
    # 2    = Latitude
    # 3    = N or S
    # 4    = Longitude
    # 5    = E or W
    # 6    = GPS quality indicator (0=invalid; 1=GPS fix; 2=Diff. GPS fix)
    # 7    = Number of satellites in use [not those in view]
    # 8    = Horizontal dilution of position
    # 9    = Antenna altitude above/below mean sea level (geoid)
    # 10   = Meters  (Antenna height unit)
    # 11   = Geoidal separation (Diff. between WGS-84 earth ellipsoid and
    #        mean sea level.  -=geoid is below WGS-84 ellipsoid)
    # 12   = Meters  (Units of geoidal separation)
    # 13   = Age in seconds since last update from diff. reference station
    # 14   = Diff. reference station ID#
    # 15   = Checksum
    def format_gga(self, utc_sec, lat, lon, fix, nsat, hdop, alt):
        fmt = "$GPGGA,%s,%s,%s,%01d,%02d,%04.1f,%07.2f,M,%07.2f,M,,"
        msg = fmt % (self.format_time(utc_sec), self.format_lat(lat), self.format_lon(lon), fix, nsat, hdop, alt, -self.nmea_settings.wgs84_to_amsl)
        return msg + "*%02X" % self.nmea_checksum(msg)

    # From: http://aprs.gids.nl/nmea/#rmc
    # eg4. $GPRMC,hhmmss.ss,A,llll.ll,a,yyyyy.yy,a,x.x,x.x,ddmmyy,x.x,a*hh
    # 1    = UTC of position fix
    # 2    = Data status (V=navigation receiver warning)
    # 3    = Latitude of fix
    # 4    = N or S
    # 5    = Longitude of fix
    # 6    = E or W
    # 7    = Speed over ground in knots
    # 8    = Track made good in degrees True
    # 9    = UT date
    # 10   = Magnetic variation degrees (Easterly var. subtracts from true course)
    # 11   = E or W
    # 12   = Checksum
    def format_rmc(self, utc_sec, fix, lat, lon, speed, course):
        fmt = "$GPRMC,%s,%s,%s,%s,%.2f,%.2f,%s,,"
        msg = fmt % (self.format_time(utc_sec), fix, self.format_lat(lat), self.format_lon(lon),
                     speed, course, self.format_date(utc_sec))
        return msg + "*%02X" % self.nmea_checksum(msg)

    def cmd_socat(self, args):
        if len(args) < 1:
            print(self.usage)
            return
        socat_cmd = args[0]
        (self.socat_in, self.socat_out) = os.pipe()
        self.socat = subprocess.Popen(["socat",
                                       "FD:0",
                                       socat_cmd],
                                      stdin=self.socat_in,
        )

    def cmd_log(self, args):
        if len(args) < 1:
            print(self.usage)
            return
        filepath = args[0]
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.mpstate.status.logdir, filepath)
        self.log_output = open(filepath, "ab")
        self.log_output_filepath = filepath

    def cmd_udp(self, args):
        if len(args) < 1:
            print(self.usage)
            return
        (hostname, port) = args[0].split(":")
        self.udp_output_port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_output_address = (hostname, int(port))
        self.udp_output_port.connect(self.udp_output_address)
        self.udp_output_port.setblocking(0)

    def set_secondary_vehicle_position(self, m):
        '''register secondary vehicle position'''
#        print("Secondary vehicle position")
        self.handle_position(m)

    def set_console_status(self, colour):
        self.console.set_status('NMEA' + str(self.instance),
                                'NMEA%u:%u' % (self.instance, self.sent_count),
                                fg=colour, row=6)

    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        import time

        if self.sent_count == 0:
            self.set_console_status('black')

        self.handle_position(m)

    def handle_position(self, m):
#        print("%u: %u vs %u" % (self.instance, m.get_srcSystem(), self.nmea_settings.target_system))
        if m.get_srcSystem() != self.nmea_settings.target_system:
            return

        now_time = time.time()
        if abs(self.output_time - now_time) < 1.0:
            return

        if m.get_type() == 'GPS_RAW_INT':
            self.num_sat = m.satellites_visible
            self.hdop = m.eph/100.0
            self.fix_quality = 1 if (m.fix_type > 1) else 0 # 0/1 for (in)valid or 2 DGPS

        if m.get_type() == 'GLOBAL_POSITION_INT':
            if self.last_time_boot_ms > m.time_boot_ms + 120000:
                # time wrap
                self.last_time_boot_ms = m.time_boot_ms
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
            altitude = m.alt * 0.001

            # for GPRMC
            speed_ms = math.sqrt(m.vx**2+m.vy**2) * 0.01
            knots = speed_ms*1.94384
            course = math.degrees(math.atan2(m.vy,m.vx))
            if course < 0:
                course += 360

            # for GPGGA

            # print format_gga(utc_sec, lat, lon, self.fix_quality, self.num_sat, self.hdop, altitude)
            # print format_rmc(utc_sec, fix_status, lat, lon, knots, course)
            gga = self.format_gga(utc_sec, lat, lon, self.fix_quality, self.num_sat, self.hdop, altitude)
            rmc = self.format_rmc(utc_sec, fix_status, lat, lon, knots, course)

            self.output_time = now_time
            #print(gga+'\r')
            #print(rmc+'\r')
            #print(self.serial)
            try:
                self.output(gga + '\r\n')
            except Exception as e:
                return
            try:
                self.output(rmc + '\r\n')
            except Exception as e:
                return
            self.sent_count += 1

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
            try:
                self.udp_output_port.send(output)
                self.set_console_status('green')
            except Exception as e:
                self.set_console_status('red')
                raise e

def init(mpstate):
    '''initialise module'''
    return NMEAModule(mpstate)
