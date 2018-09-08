'''class to interpolate position information given a time'''

import sys, os, time, math, datetime, re
import fractions

from MAVProxy.modules.mavproxy_map import mp_elevation
from pymavlink import mavutil
from cuav.lib import cuav_util

EleModel = mp_elevation.ElevationModel()

class MavInterpolatorException(Exception):
    '''interpolator error class'''
    def __init__(self, msg):
            Exception.__init__(self, msg)

class MavInterpolatorDeltaTException(MavInterpolatorException):
    '''interpolator error class for over deltat'''
    def __init__(self, msg):
            Exception.__init__(self, msg)

class MavPosition():
    '''represent current position and attitude
    The height is in meters above home ground level
    '''
    def __init__(self, lat, lon, altitude, roll, pitch, yaw, frame_time=None):
        self.lat = lat
        self.lon = lon
        self.altitude = altitude
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.time = frame_time

    def __str__(self):
        return 'MavPosition(pos %f %f alt=%.1f roll=%.1f pitch=%.1f yaw=%.1f)' % (
            self.lat, self.lon, self.altitude,
            self.roll, self.pitch, self.yaw)

class JitterCorrection():
    def __init__(self):
        self.max_lag_s = 60.0
        self.convergence_loops = 0
        self.link_offset_s = 0.0
        self.min_sample_s = 0.0
        self.initialised = False
        self.min_sample_counter = 0
        self.last_corrected_s = None

    def correct_local(self, local_s):
        if self.last_corrected_s is None:
            return local_s
        return self.last_corrected_s
    
    def correct_timestamp(self, offboard_s, local_s):
        '''correct an offboard timestamp into local time'''
        diff_s = local_s - offboard_s

        if not self.initialised or diff_s < self.link_offset_s:
            '''this message arrived from the remote system with a timestamp
               that would imply the message was from the future. We know that
               isn't possible, so we adjust down the correction value'''
            self.link_offset_s = diff_s;
            #print("link_offset_s=%f" % self.link_offset_s)
            self.initialised = True

        estimate_s = offboard_s + self.link_offset_s
        if estimate_s > local_s:
            # this should be impossible, just check it under SITL
            printf("ERR: msg from future %f" % (estimate_s - local_s))

        if estimate_s + self.max_lag_s < local_s:
            '''this implies the message came from too far in the past. Clamp
            the lag estimate to assume the message had maximum lag'''
            print("ERR: offboard timestamp too old %f" % (local_s - estimate_s))
            estimate_s = local_s - self.max_lag_s
            self.link_offset_s = estimate_s - offboard_s

        if self.min_sample_counter == 0:
            self.min_sample_s = diff_s

        self.min_sample_counter += 1
        if diff_s < self.min_sample_s:
            self.min_sample_s = diff_s

        if self.min_sample_counter == 1000:
            '''we have 1000 samples of the transport lag. To account for long
            term clock drift we set the diff we will use in future to this
            value '''
            self.link_offset_s = self.min_sample_s
            self.min_sample_counter = 0
            #print("new link_offset_s=%f" % self.min_sample_s)

        if self.last_corrected_s is not None and estimate_s < self.last_corrected_s:
            # don't allow time to go backwards
            estimate_s = self.last_corrected_s
        self.last_corrected_s = estimate_s
        return estimate_s
    
    
class MavInterpolator():
    '''a class to interpolate position and attitude from a
    series of mavlink messages'''
    def __init__(self, backlog=500, gps_lag=0):
        self.backlog = backlog
        self.attitude = []
        self.global_position_int = []
        self.terrain_report = []
        self.msg_map = {
            'GLOBAL_POSITION_INT' : self.global_position_int,
            'ATTITUDE' : self.attitude,
            'TERRAIN_REPORT' : self.terrain_report
            }
        self.mlog = None
        self.ground_pressure = None
        self.usec_base = 0
        self.boot_offset = 0
        self.last_msg_time = 0
        self.gps_lag = gps_lag
        self.jitter = JitterCorrection()
        self.jitter_correction = True

    def _find_msg_idx(self, type, t):
        '''find the msg just before time t'''
        if not type in self.msg_map:
            raise MavInterpolatorException('no msgs of type %s' % type)
        a = self.msg_map[type]
        for i in range(len(a)-1, -1, -1):
            if a[i]._timestamp <= t:
                return i
        if len(a) > 0:
            last_timestamp = time.asctime(time.localtime(a[-1]._timestamp))
        else:
            last_timestamp = ''
        raise MavInterpolatorException('no msgs of type %s before %s last=%s' % (
            type, time.asctime(time.localtime(t)), last_timestamp))

    def _find_msg(self, type, t):
        '''find the msg just before time t'''
        if not type in self.msg_map:
            raise MavInterpolatorException('no msgs of type %s' % type)
        i = self._find_msg_idx(type, t)
        return self.msg_map[type][i]


    def add_msg(self, msg):
        '''add in a mavlink message'''
        type = msg.get_type()
        if type in self.msg_map:
            '''add it to the history'''
            self.msg_map[type].append(msg)
            '''keep self.backlog messages around of each type'''
            while len(self.msg_map[type]) > self.backlog:
                self.msg_map[type].pop(0)
        if self.jitter_correction:
            if type in ['ATTITUDE', 'GLOBAL_POSITION_INT']:
                timestamp_corrected = self.jitter.correct_timestamp(msg.time_boot_ms*0.001, msg._timestamp)
                #print(msg._timestamp - timestamp_corrected)
                msg._timestamp = timestamp_corrected
            else:
                msg._timestamp = self.jitter.correct_local(msg._timestamp)
            
    def _altitude(self, GLOBAL_POSITION_INT, TERRAIN_REPORT):
        '''get height above the ground'''
        if TERRAIN_REPORT is not None:
            return TERRAIN_REPORT.current_height
        return GLOBAL_POSITION_INT.relative_alt*0.001

    def advance_log(self, t):
        '''read from the logfile to advance to time t'''
        if self.mlog is None:
            return
        while True:
            try:
                raw = 'GLOBAL_POSITION_INT'
                gps_raw = self._find_msg(raw, t)
                attitude = self._find_msg('ATTITUDE', t)
                if (self.msg_map[raw][-1]._timestamp >= t and
                    self.msg_map['ATTITUDE'][-1]._timestamp >= t):
                    return
            except MavInterpolatorException:
                pass
            msg = self.mlog.recv_match()
            if msg is None:
                return MavInterpolatorException('end of logfile for timestamp %s' % time.asctime(time.localtime(t)))
            self.add_msg(msg)
            if msg._timestamp > t+3:
                break

    def interpolate(self, type, field, t, max_deltat=0):
        '''find interpolated value for a field'''
        i = self._find_msg_idx(type, t)
        a = self.msg_map[type]
        if i == len(a)-1:
            return getattr(a[i], field)
        v1 = getattr(a[i], field)
        v2 = getattr(a[i+1], field)
        t1 = a[i]._timestamp
        t2 = a[i+1]._timestamp
        if max_deltat != 0 and t2 - t1 > max_deltat:
            raise MavInterpolatorDeltaTException('exceeded max_deltat %.1f' % (t2-t1))
        return v1 + ((t-t1)/(t2-t1))*(v2-v1)

    def interpolate_angle(self, type, field, t, max_deltat=0):
        '''find interpolated value for a angle field in range -pi to pi'''
        i = self._find_msg_idx(type, t)
        a = self.msg_map[type]
        if i == len(a)-1:
            return getattr(a[i], field)
        v1 = getattr(a[i], field)
        v2 = getattr(a[i+1], field)
        if abs(v1 - v2) > math.pi:
            if v1 < v2:
                v1 += 2*math.pi
            else:
                v2 += 2*math.pi
        t1 = a[i]._timestamp
        t2 = a[i+1]._timestamp
        if max_deltat != 0 and t2 - t1 > max_deltat:
            raise MavInterpolatorDeltaTException('exceeded max_deltat %.1f' % (t2-t1))
        ret = v1 + ((t-t1)/(t2-t1))*(v2-v1)
        if ret > math.pi:
            ret -= 2*math.pi
        #print(t1, t2, t, math.degrees(v1), math.degrees(v2), math.degrees(ret))
        return ret

    # maxroll and maxpitch represent the maximum roll and pitch
    # that can be stabilised by the stabilisation system
    def position(self, t, max_deltat=0,roll=None, pitch=None, maxroll=0, maxpitch=0, pitch_offset=0, roll_offset=0):
        '''return a MavPosition estimate given a time'''
        self.advance_log(t)

        # interpolate our latitude/longitude
        lat = self.interpolate('GLOBAL_POSITION_INT', 'lat', t, max_deltat)*1.0e-7
        lon = self.interpolate('GLOBAL_POSITION_INT', 'lon', t, max_deltat)*1.0e-7

        terrain_report = None
        if len(self.terrain_report) > 0:
            terrain_report = self._find_msg('TERRAIN_REPORT', t)
            if terrain_report is not None:
                (tlat, tlon) = (terrain_report.lat/1.0e7, terrain_report.lon/1.0e7)
                # don't use it if its too far away
                if (cuav_util.gps_distance(lat, lon, tlat, tlon) > 150 or
                    abs(terrain_report._timestamp - t) > 5):
                        terrain_report = None

        # get altitude
        gpst = t + self.gps_lag
        gps_pos = self._find_msg('GLOBAL_POSITION_INT', gpst)
        altitude = self._altitude(gps_pos, terrain_report)

        # and attitude
        if roll is None:
            roll = math.degrees(self.interpolate_angle('ATTITUDE', 'roll', t, max_deltat))
        if abs(roll) < maxroll:
                        # camera stabilisation system can take care of it
            roll = 0
        elif roll >= maxroll:
                        # adjust for roll stabilisation system can't handle
            roll = roll - maxroll
        else:
                        # adjust for roll stabilisation system can't handle
            roll = roll + maxroll

        if pitch is None:
            pitch = math.degrees(self.interpolate_angle('ATTITUDE', 'pitch', t, max_deltat))
        if abs(pitch) < maxpitch:
            pitch = 0
        elif pitch >= maxpitch:
            pitch = pitch - maxpitch
        else:
            pitch = pitch + maxpitch

        # add pitch and roll offset
        pitch += pitch_offset
        roll += roll_offset

        yaw   = math.degrees(self.interpolate_angle('ATTITUDE', 'yaw', t, max_deltat))

        return MavPosition(lat, lon, altitude, roll, pitch, yaw, t)
    
    def set_logfile(self, filename):
        '''provide a mavlink logfile for data'''
        self.mlog = mavutil.mavlogfile(filename)
        

class Fraction(fractions.Fraction):
    """Only create Fractions from floats.

    >>> Fraction(0.3)
    Fraction(3, 10)
    >>> Fraction(1.1)
    Fraction(11, 10)
    """

    def __new__(cls, value, ignore=None):
        """Should be compatible with Python 2.6, though untested."""
        return fractions.Fraction.from_float(value).limit_denominator(99999)

def dms_to_decimal(degrees, minutes, seconds, sign=' '):
    """Convert degrees, minutes, seconds into decimal degrees.

    >>> dms_to_decimal(10, 10, 10)
    10.169444444444444
    >>> dms_to_decimal(8, 9, 10, 'S')
    -8.152777777777779
    """
    return (-1 if sign[0] in 'SWsw' else 1) * (
        float(degrees)        +
        float(minutes) / 60   +
        float(seconds) / 3600
    )

def decimal_to_dms(decimal):
    """Convert decimal degrees into degrees, minutes, seconds.

    >>> decimal_to_dms(50.445891)
    [Fraction(50, 1), Fraction(26, 1), Fraction(113019, 2500)]
    >>> decimal_to_dms(-125.976893)
    [Fraction(125, 1), Fraction(58, 1), Fraction(92037, 2500)]
    """
    remainder, degrees = math.modf(abs(decimal))
    remainder, minutes = math.modf(remainder * 60)
    return [Fraction(n) for n in (degrees, minutes, remainder * 60)]

_last_position = None

def exif_position(filename):
        '''get a MavPosition from exif tags

        See: http://stackoverflow.com/questions/10799366/geotagging-jpegs-with-pyexiv2
        '''
        import pyexiv2
        global _last_position
        
        m = pyexiv2.ImageMetadata(filename)
        m.read()
        GPS = 'Exif.GPSInfo.GPS'
        try:
                lat_ns = str(m[GPS + 'LatitudeRef'].value)
                lng_ns = str(m[GPS + 'LongitudeRef'].value)
                latitude = dms_to_decimal(m[GPS + 'Latitude'].value[0],
                                          m[GPS + 'Latitude'].value[1],
                                          m[GPS + 'Latitude'].value[2],
                                          lat_ns)
                longitude = dms_to_decimal(m[GPS + 'Longitude'].value[0],
                                           m[GPS + 'Longitude'].value[1],
                                           m[GPS + 'Longitude'].value[2],
                                           lng_ns)
        except Exception:
                latitude = 0
                longitude = 0
                

        altitude = float(m[GPS + 'Altitude'].value)

        timestamp = (os.path.splitext(os.path.basename(filename))[0])
        m = re.search("\d", timestamp)
        if m :
            timestamp = timestamp[m.start():]
        
        frame_time = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S%fZ")
        frame_time = cuav_util.datetime_to_float(frame_time)
        
        if _last_position is None:
                yaw = 0
        else:
                yaw = cuav_util.gps_bearing(_last_position.lat, _last_position.lon,
                                            latitude, longitude)
        pos = MavPosition(latitude, longitude, altitude, 0, 0, yaw, frame_time)
        _last_position = pos
        return pos


class KmlPosition(object):
        '''parse a kmz file to get positions for images'''
        def __init__(self, filenames):
                import glob
                self.images = {}
                for f in glob.glob(filenames):
                        self._add_kmz(f)

        def _add_kmz(self, filename):
                import xml.dom.minidom
                dom = None
                if filename.endswith('.kmz'):
                        import zipfile
                        z = zipfile.ZipFile(filename, mode='r')
                        names = z.namelist()
                        for n in names:
                                if n.endswith('.kml'):
                                        dom = xml.dom.minidom.parse(z.open(n))
                                        break
                else:
                        dom = xml.dom.minidom.parse(filename)
                if dom is None:
                        return
                marks = dom.getElementsByTagName('Placemark')
                for m in marks:
                    try:
                        name = self._getElement(m.getElementsByTagName('name')[0])
                        coords = (m.getElementsByTagName('Point')[0].getElementsByTagName('coordinates')[0]).childNodes
                        coordsstr = str(self.getText(coords))
                        latitude = coordsstr.split(',')[1]
                        longitude = coordsstr.split(',')[0]
                        self.images[name] = MavPosition(float(latitude), float(longitude), 0, 0, 0, 0, 0)
                    except:
                        pass

        def getText(self, nodelist):
            rc = []
            for node in nodelist:
                if node.nodeType == node.TEXT_NODE:
                    rc.append(node.data)
            return ''.join(rc)
    
        def position(self, imagename):
                imagename = os.path.basename(imagename)
                if not imagename in self.images:
                        print("No position for %s" % imagename)
                        return self.images[0]
                return self.images[imagename]

        def _getText(self, nodelist):
                '''Get the text inside an XML node'''
                rc = ""
                for node in nodelist:
                        if node.nodeType == node.TEXT_NODE:
                                rc = rc + node.nodeValue
                return rc
        
        def _getElement(self, element):
                '''Get and XML element'''
                return self._getText(element.childNodes)


class TriggerPosition(object):
        '''parse a Robota trigger file to get positions for images'''
        def __init__(self, filename):
                f = open(filename)
                lines = f.readlines()
                f.close()
                self.positions = []
                self.columns = lines[0].rstrip().split(' ')
                self.colmap = {}
                self.time_offset = None
                for i in range(len(self.columns)):
                        self.colmap[self.columns[i]] = i
                for i in range(1, len(lines)):
                        self._parse_line(lines[i].rstrip())

        def _column(self, colname, vals, defvalue):
                '''return a value for a column'''
                import datetime
                if not colname in self.colmap:
                        return defvalue
                v = vals[self.colmap[colname]]
                if v.endswith('Z'):
                        v = time.mktime(datetime.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").timetuple())
                else:
                        v = float(v)
                return v

        def _parse_line(self, line):
                '''parse one line'''
                vals = line.split(' ')
                lat = self._column('Lat(deg)', vals, 0)
                lon = self._column('Lon(deg)', vals, 0)
                ground_alt = get_ground_alt(lat, lon)
                pos = MavPosition(lat, lon,
                                  max(self._column('GpsAlt(m)', vals, 0) - ground_alt, 10),
                                  self._column('Roll(deg)', vals, 0),
                                  self._column('Pitch(deg)', vals, 0),
                                  self._column('Heading(deg)', vals, 0),
                                  self._column('DateTimeYYYY-MM-DDTHH:MM:SSZ', vals, 0))
                self.positions.append(pos)                                  

        def position(self, imagename):
                '''find the best matching MavPosition for an image file'''
                timestamp = (os.path.splitext(os.path.basename(imagename))[0])
                m = re.search("\d", timestamp)
                if m :
                    timestamp = timestamp[m.start():]
                
                frame_time = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S%fZ")
                frame_time = cuav_util.datetime_to_float(frame_time)
                
                if self.time_offset is None:
                        # assume first file matches first trigger record
                        self.time_offset = frame_time - self.positions[0].time
                # find the best time match
                besti = 0
                bestdt = -1
                for i in range(len(self.positions)):
                        dt = abs((self.positions[i].time - frame_time) + self.time_offset)
                        if bestdt == -1 or dt < bestdt:
                                bestdt = dt
                                besti = i
                return self.positions[besti]

def get_ground_alt(lat, lon):
    '''get highest ground altitide around a point'''
    global EleModel
    ground = EleModel.GetElevation(lat, lon)
    surrounds = []
    for bearing in range(0, 360, 45):
        surrounds.append((150, bearing))
    for (dist, bearing) in surrounds:
        (lat2, lon2) = cuav_util.gps_newpos(lat, lon, bearing, dist)
        el = EleModel.GetElevation(lat2, lon2)
        if el > ground:
            ground = el
    return ground

