'''common utility functions'''

import numpy, cv, math, sys

class PGMError(Exception):
	'''PGMLink error class'''
	def __init__(self, msg):
            Exception.__init__(self, msg)


class PGM(object):
    '''16 bit 1280x960 PGM image handler'''
    def __init__(self, filename):
        self.filename = filename
        
        f = open(filename, mode='r')
        fmt = f.readline()
        if fmt.strip() != 'P5':
            raise PGMError('Expected P5 image in %s' % filename)
        dims = f.readline()
        if dims.strip() != '1280 960':
            raise PGMError('Expected 1280x960 image in %s' % filename)
        line = f.readline()
        if line[0] == '#':
            self.comment = line
            line = f.readline()
        if line.strip() != '65535':
            raise PGMError('Expected 16 bit image image in %s - got %s' % (filename, line.strip()))
        ofs = f.tell()
        f.close()
        a = numpy.memmap(filename, dtype='uint16', mode='c', order='C', shape=(960,1280), offset=ofs)
        self.array = a.byteswap(True)
        self.img = cv.CreateImageHeader((1280, 960), 16, 1)
        cv.SetData(self.img, self.array.tostring(), self.array.dtype.itemsize*1*1280)

def key_menu(i, n, image, filename):
    '''simple keyboard menu'''
    while True:
        key = cv.WaitKey()
        if not key in range(128):
            continue
        key = chr(key)
        if key == 'q':
            sys.exit(0)
        if key == 's':
            print("Saving %s" % filename)
            cv.SaveImage(filename, image)
        if key in ['n', '\n', ' ']:
            if i == n-1:
                print("At last image")
            else:
                return i+1
        if key == 'b':
            if i == 0:
                print("At first image")
            else:
                return i-1

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


def ground_offset(height, pitch, roll, yaw):
    '''
    find the offset on the ground in meters of the center of view of the plane
    given height above the ground in meters, and pitch/roll/yaw in degrees.

    The yaw is from grid north. Positive yaw is clockwise
    The roll is from horiznotal. Positive roll is down on the right
    The pitch is from horiznotal. Positive pitch is up in the front

    return result is a tuple, with meters east and north of GPS position

    This is only correct for small values of pitch/roll
    '''

    # x/y offsets assuming the plan is pointing north
    xoffset = height * math.tan(math.radians(roll))
    yoffset = height * math.tan(math.radians(pitch))
    
    # convert to polar coordinates
    distance = math.hypot(xoffset, yoffset)
    angle    = math.atan2(yoffset, xoffset)

    # add in yaw
    angle -= math.radians(yaw)

    # back to rectangular coordinates
    x = distance * math.cos(angle)
    y = distance * math.sin(angle)

    return (x, y)


def pixel_position(xpos, ypos, height, pitch, roll, yaw,
                   lens=4.0, sensorwidth=5.0, xresolution=1280, yresolution=960):
    '''
    find the offset on the ground in meters of a pixel in a ground image
    given height above the ground in meters, and pitch/roll/yaw in degrees, the
    lens and image parameters

    The xpos,ypos is from the top-left of the image
    The height is in meters
    
    The yaw is from grid north. Positive yaw is clockwise
    The roll is from horiznotal. Positive roll is down on the right
    The pitch is from horiznotal. Positive pitch is up in the front
    lens is in mm
    sensorwidth is in mm
    xresolution and yresolution is in pixels
    
    return result is a tuple, with meters east and north of current GPS position

    This is only correct for small values of pitch/roll
    '''
    
    (xcenter, ycenter) = ground_offset(height, pitch, roll, yaw)
    
    px = pixel_width(height, xresolution=xresolution, lens=lens, sensorwidth=sensorwidth)
    py = pixel_height(height, yresolution=yresolution, lens=lens, sensorwidth=sensorwidth)

    dx = (xresolution/2) - xpos
    dy = (yresolution/2) - ypos

    range_c = math.hypot(dx * px, dy * py)
    angle_c = math.atan2(dy * py, dx * px)

    # add in yaw
    angle_c += math.radians(yaw)

    # back to rectangular coordinates
    x = - range_c * math.cos(angle_c)
    y = range_c * math.sin(angle_c)

    return (xcenter+x, ycenter+y)


def pixel_coordinates(xpos, ypos, latitude, longitude, height, pitch, roll, yaw,
                      lens=4.0, sensorwidth=5.0, xresolution=1280, yresolution=960):
    '''
    find the latitude/longitude of a pixel in a ground image given
    our GPS position, our height above the ground in meters, and pitch/roll/yaw in degrees,
    the lens and image parameters

    The xpos,ypos is from the top-left of the image
    latitude is in degrees. Negative for south
    longitude is in degrees
    The height is in meters
    
    The yaw is from grid north. Positive yaw is clockwise
    The roll is from horiznotal. Positive roll is down on the right
    The pitch is from horiznotal. Positive pitch is up in the front
    lens is in mm
    sensorwidth is in mm
    xresolution and yresolution is in pixels
    
    return result is a tuple, with meters east and north of current GPS position

    This is only correct for small values of pitch/roll
    '''

    (xofs, yofs) = pixel_position(xpos, ypos, height, pitch, roll, yaw,
                                  lens=lens, sensorwidth=sensorwidth,
                                  xresolution=xresolution, yresolution=yresolution)

    radius_of_earth = 6378100.0 # in meters

    dlat = math.degrees(math.atan(yofs/radius_of_earth))
    dlon = math.degrees(math.atan(xofs/radius_of_earth))

    return (latitude+dlat, longitude+dlon)
