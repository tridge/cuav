#!/usr/bin/env python
'''common CanberraUAV utility functions'''

import numpy, cv2, math, sys, os, time, struct, calendar, re, datetime

import six; from six.moves import cPickle as pickle
from cuav.camera.cam_params import CameraParams
from . import rotmat

radius_of_earth = 6378100.0 # in meters


def gps_distance(lat1, lon1, lat2, lon2):
    '''return distance between two points in meters,
    coordinates are in degrees
    thanks to http://www.movable-type.co.uk/scripts/latlong.html'''
    from math import radians, cos, sin, sqrt, atan2
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    
    a = sin(0.5*dLat)**2 + sin(0.5*dLon)**2 * cos(lat1) * cos(lat2)
    c = 2.0 * atan2(sqrt(a), sqrt(1.0-a))
    return radius_of_earth * c


def gps_bearing(lat1, lon1, lat2, lon2):
    '''return bearing between two points in degrees, in range 0-360
    thanks to http://www.movable-type.co.uk/scripts/latlong.html'''
    from math import sin, cos, atan2, radians, degrees
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    dLat = lat2 - lat1
    dLon = lon2 - lon1    
    y = sin(dLon) * cos(lat2)
    x = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dLon)
    bearing = degrees(atan2(y, x))
    if bearing < 0:
        bearing += 360.0
    return bearing


def gps_newpos(lat, lon, bearing, distance):
    '''extrapolate latitude/longitude given a heading and distance 
    thanks to http://www.movable-type.co.uk/scripts/latlong.html
    '''
    from math import sin, asin, cos, atan2, radians, degrees

    lat1 = radians(lat)
    lon1 = radians(lon)
    brng = radians(bearing)
    dr = distance/radius_of_earth

    lat2 = asin(sin(lat1)*cos(dr) +
            cos(lat1)*sin(dr)*cos(brng))
    lon2 = lon1 + atan2(sin(brng)*sin(dr)*cos(lat1), 
                cos(dr)-sin(lat1)*sin(lat2))
    return (degrees(lat2), degrees(lon2))



def angle_of_view(lens=4.0, sensorwidth=5.0):
    '''
    return angle of view in degrees of the lens

    sensorwidth is in millimeters
    lens is in mm
    '''
    return math.degrees(2.0*math.atan((sensorwidth/1000.0)/(2.0*lens/1000.0)))

def groundwidth(height, lens=4.0, sensorwidth=5.0):
    '''
    return frame width on ground in meters

    height is in meters
    sensorwidth is in millimeters
    lens is in mm
    '''
    aov = angle_of_view(lens=lens, sensorwidth=sensorwidth)
    return 2.0*height*math.tan(math.radians(0.5*aov))


def pixel_width(height, xresolution=1280, lens=4.0, sensorwidth=5.0):
    '''
    return pixel width on ground in meters

    height is in meters
    xresolution is in pixels
    lens is in mm
    sensorwidth is in mm
    '''
    return groundwidth(height, lens=lens, sensorwidth=sensorwidth)/xresolution

def pixel_height(height, yresolution=960, lens=4.0, sensorwidth=5.0):
    '''
    return pixel height on ground in meters

    height is in meters
    yresolution is in pixels
    lens is in mm
    sensorwidth is in mm
    '''
    return groundwidth(height, lens=lens, sensorwidth=sensorwidth)/yresolution


def pixel_position_matt(xpos, ypos, height, pitch, roll, yaw, C):
    '''
    find the offset on the ground in meters of a pixel in a ground image
    given height above the ground in meters, and pitch/roll/yaw in degrees, the
    lens and image parameters

    The xpos,ypos is from the top-left of the image
    The height is in meters
    
    The yaw is from grid north. Positive yaw is clockwise
    The roll is from horiznotal. Positive roll is down on the right
    The pitch is from horiznotal. Positive pitch is up in the front
    
    return result is a tuple, with meters east and north of current GPS position

    '''
    from numpy import array,eye, zeros, uint64
    from cuav.uav.uav import uavxfer
    from math import pi
  
    xfer = uavxfer()
    xfer.setCameraMatrix(C.K)
    xfer.setCameraOrientation( 0.0, 0.0, pi/2 )
    xfer.setFlatEarth(0);
    xfer.setPlatformPose(0, 0, -height, math.radians(roll), math.radians(pitch), math.radians(yaw))

    # compute the undistorted points for the ideal camera matrix
    #src = numpy.zeros((1,1,2), numpy.uint64) #cv.CreateMat(1, 1, cv.CV_64FC2)
    src = numpy.zeros((1,1, 2), numpy.float32)
    src[0,0] = (xpos, ypos)
    #dst = cv.CreateMat(1, 1, cv.CV_64FC2)
    R = eye(3)
    K = C.K
    D = C.D
    dst = cv2.undistortPoints(src, K, D, R, K)
    x = dst[0,0][0]
    y = dst[0,0][1]
    #print '(', xpos,',', ypos,') -> (', x, ',', y, ')'
    # negative scale means camera pointing above horizon
    # large scale means a long way away also unreliable
    (joe_w, scale) = xfer.imageToWorld(x, y)
    if (scale < 0 or scale > 500):
      return None

    #(te, tn) = pixel_position_tridge(xpos, ypos, height, pitch, roll, yaw, lens, sensorwidth, xresolution, yresolution)
    #diff = (te-joe_w[1], tn-joe_w[0])
    #print 'diff: ', diff

    # east and north
    return (joe_w[1], joe_w[0])


def pixel_coordinates(xpos, ypos, latitude, longitude, height, pitch, roll, yaw, C):
    '''
    find the latitude/longitude of a pixel in a ground image given
    our GPS position, our height above the ground in meters, and pitch/roll/yaw in degrees,
    the lens and image parameters

    The xpos,ypos is from the top-left of the image
    latitude is in degrees. Negative for south
    longitude is in degrees
    The height is in meters
    
    The yaw is from grid north. Positive yaw is clockwise
    The roll is from horizontal. Positive roll is down on the right
    The pitch is from horizontal. Positive pitch is up in the front
    
    return result is a tuple, with meters east and north of current GPS position

    This is only correct for small values of pitch/roll
    '''

    
    pt = pixel_position_matt(xpos, ypos, height, pitch, roll, yaw, C)
    if pt is None:
        # its pointing into the sky
        return None
    (xofs, yofs) = pt

    bearing = math.degrees(math.atan2(xofs, yofs))
    distance = math.sqrt(xofs**2 + yofs**2)
    return gps_newpos(latitude, longitude, bearing, distance)

def gps_position_from_xy(x, y, pos, C=None, altitude=None, shape=None):
    '''
    return a GPS position in an image given a MavPosition object
    and an image x,y position
    '''
    if C is None:
            raise ValueError("camera parameters must be supplied")
    if pos is None:
            return None
    if shape is not None:
            (width,height) = shape
            # assume the image came from the same camera but may no longer be original size
            scale_x = float(C.xresolution)/float(width)
            scale_y = float(C.yresolution)/float(height)
            x *= scale_x
            y *= scale_y
    if altitude is None:
        altitude = pos.altitude
    return pixel_coordinates(x, y, pos.lat, pos.lon, altitude,
                             pos.pitch, pos.roll, pos.yaw, C)

def meters_per_pixel(pos, C):
        '''return meters per pixel scale given a MavPosition'''
        width=C.xresolution
        height=C.yresolution
        p1 = gps_position_from_xy(0, height/2, pos, C=C)
        p2 = gps_position_from_xy(width-1, height/2, pos, C=C)
        if p1 is None or p2 is None:
                return None
        dist = gps_distance(p1[0], p1[1], p2[0], p2[1])
        mpp = dist / float(width)
        return mpp
        

def gps_position_from_image_region(region, pos, width=1280, height=960, C=None, altitude=None):
    '''
    return a GPS position in an image given a MavPosition object
    and an image region tuple
    '''
    if C is None:
        raise ValueError("camera parameters must be supplied")
    if pos is None:
        return None
    x = (region.x1+region.x2)*0.5
    y = (region.y1+region.y2)*0.5
    return gps_position_from_xy(x, y, pos, C=C, shape=(width,height), altitude=altitude)

def mkdir_p(dir):
    '''like mkdir -p'''
    if not dir:
        return
    if dir.endswith("/"):
        mkdir_p(dir[:-1])
        return
    if os.path.isdir(dir):
        return
    mkdir_p(os.path.dirname(dir))
    try:
        os.mkdir(dir)
    except Exception:
        pass

def frame_time(t):
    '''return a time string for a filename with 0.01 sec resolution'''
    # round to the nearest 100th of a second
    t += 0.005
    hundredths = int(t * 100.0) % 100
    return "%s%02uZ" % (time.strftime("%Y%m%d%H%M%S", time.gmtime(t)), hundredths)

def datetime_to_float(d):
    """Datetime object to seconds since epoch (float)"""
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds

def parse_frame_time(filename):
    '''parse a image frame time from a image filename
    '''
    timestamp = (os.path.splitext(os.path.basename(filename))[0])
    m = re.search("\d", timestamp)
    if m :
        timestamp = timestamp[m.start():]
    
    frame_time = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S%fZ")
    t = datetime_to_float(frame_time)
    
    return t


def polygon_outside(P, V):
    '''return true if point is outside polygon
    P is a (x,y) tuple
    V is a list of (x,y) tuples

    The point in polygon algorithm is based on:
    http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
    '''
    n = len(V)
    outside = True
    j = n-1
    for i in range(n):
        if (((V[i][1]>P[1]) != (V[j][1]>P[1])) and
            (P[0] < (V[j][0]-V[i][0]) * (P[1]-V[i][1]) / (V[j][1]-V[i][1]) + V[i][0])):
            outside = not outside
        j = i
    return outside


def polygon_load(filename):
    '''load a polygon from a file'''
    ret = []
    f = open(filename)
    for line in f:
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            continue
        a = line.split()
        if len(a) != 2:
            raise RuntimeError("invalid polygon line: %s" % line)
        ret.append((float(a[0]), float(a[1])))
    f.close()
    return ret


def polygon_complete(V):
    '''
    check if a polygon is complete. 

    We consider a polygon to be complete if we have at least 4 points,
    and the first point is the same as the last point. That is the
    minimum requirement for the polygon_outside function to work
    '''
    return (len(V) >= 4 and V[-1][0] == V[0][0] and V[-1][1] == V[0][1])


def image_shape(img):
    '''return (w,h) of an image, coping with different image formats'''
    height, width = img.shape[:2]
    return (width, height)

def image_width(img):
    '''return width of an image, coping with different image formats'''
    if getattr(img, 'shape', None) is not None:
        return img.shape[1]
    return getattr(img, 'width')


def SubImage(src, region):
    '''return a subimage as a new image. This allows
    for the region going past the edges.
    region is of the form (x1,y1,width,height)'''
    (x1,y1,width,height) = region
    #if src == None:
    #    return numpy.zeros((height,width,3),dtype=numpy.uint16)
    ret = numpy.zeros((height,width,3),dtype=src.dtype)
    (img_width,img_height) = image_shape(src)
    if x1 < 0:
        sx1 = 0
        xofs = -x1
    else:
        sx1 = x1
        xofs = 0
    if y1 < 0:
        sy1 = 0
        yofs = -y1
    else:
        sy1 = y1
        yofs = 0
    if sx1 + width <= img_width:
        w = width
    else:
        w = img_width - sx1
    if sy1 + height <= img_height:
        h = height
    else:
        h = img_height - sy1
    if yofs+h > height:
        h = h - yofs
    if xofs+w > width:
        w = w - xofs
        
    ret[yofs:yofs+h, xofs:xofs+w] = src[sy1:sy1+h, sx1:sx1+w]
    return ret

def OverlayImage(img, img2, x, y):
    '''overlay a 2nd image on a first image, at position x,y
    on the first image'''
    (img_width,img_height) = image_shape(img2)
    img[y:y+img_height, x:x+img_width] = img2


def SaturateImage(img, scale=1, brightness=2):
    '''return a zoomed saturated image. Assumes a RGB image'''
    (w,h) = image_shape(img)
    (w2,h2) = (w//scale, h//scale)
    img2 = cv2.resize(img, (0,0), fx=scale, fy=scale)
    hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    s[:] = 255
    v = numpy.where(v <= 255 - (brightness * 10), v + (brightness * 10), 255)
    hsv[:, :, 2] = v
    hsv[:, :, 1] = s
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def set_system_clock(time_seconds):
    '''sync system clock with GPS time
    Thanks to tMC http://stackoverflow.com/questions/12081310/python-module-to-change-system-date-and-time
    '''
    import ctypes
    import ctypes.util

    # define CLOCK_REALTIME 0
    CLOCK_REALTIME = 0

    # /usr/include/time.h
    #
    # struct timespec
    #  {
    #    __time_t tv_sec;            /* Seconds.  */
    #    long int tv_nsec;           /* Nanoseconds.  */
    #  };
    class timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), # hmm, is c_long right? what about 64bit time_t?
                    ("tv_nsec", ctypes.c_long)]
        
    librt = ctypes.CDLL(ctypes.util.find_library("rt"))

    ts = timespec()
    ts.tv_sec = int(time_seconds)
    ts.tv_nsec = int((time_seconds - int(time_seconds))*1e9)

    # http://linux.die.net/man/3/clock_settime
    return librt.clock_settime(CLOCK_REALTIME, ctypes.byref(ts))
