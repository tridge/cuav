#!/usr/bin/env python
'''common CanberraUAV utility functions'''

import numpy, cv, math, sys, os, time, rotmat, cStringIO, cPickle, struct

from cuav.camera.cam_params import CameraParams

radius_of_earth = 6378100.0 # in meters

class PGMError(Exception):
	'''PGMLink error class'''
	def __init__(self, msg):
            Exception.__init__(self, msg)


class PGM(object):
	'''8/16 bit 1280x960 PGM image handler'''
	def __init__(self, filename):
		self.filename = filename

		f = open(filename, mode='rb')
		fmt = f.readline()
		if fmt.strip() != 'P5':
			raise PGMError('Expected P5 image in %s' % filename)
		dims = f.readline()
		if dims.strip() != '1280 960':
			raise PGMError('Expected 1280x960 image in %s' % filename)
		line = f.readline()
		self.comment = None
		if line[0] == '#':
			self.comment = line
			line = f.readline()
		line = line.strip()
		if line == "65535":
			self.eightbit = False
		elif line == "255":
			self.eightbit = True
		else:
			raise PGMError('Expected 8/16 bit image image in %s - got %s' % (filename, line))
		ofs = f.tell()
		if self.eightbit:
			rawdata = numpy.fromfile(f, dtype='uint8')
			rawdata = numpy.reshape(rawdata, (960,1280))
			self.img = cv.CreateImageHeader((1280, 960), 8, 1)
		else:
			rawdata = numpy.fromfile(f, dtype='uint16')
			rawdata = rawdata.byteswap(True)
			rawdata = numpy.reshape(rawdata, (960,1280))
			self.img = cv.CreateImageHeader((1280, 960), 8, 1)
		f.close()
		self.rawdata = rawdata
		self.array = self.rawdata.byteswap(True)
		cv.SetData(self.img, self.array.tostring(), self.array.dtype.itemsize*1*1280)

def key_menu(i, n, image, filename, pgm=None):
    '''simple keyboard menu'''
    while True:
        key = cv.WaitKey()
	key &= 0xFF
        if not key in range(128):
            continue
        key = chr(key)
        if key == 'q':
            sys.exit(0)
        if key == 's':
            print("Saving %s" % filename)
            cv.SaveImage(filename, image)
        if key == 'c' and pgm is not None:
            print("Comment: %s" % pgm.comment)
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

    # x/y offsets assuming the plane is pointing north
    xoffset = -height * math.tan(math.radians(roll))
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


def pixel_position_old(xpos, ypos, height, pitch, roll, yaw,
                   lens=4.0, sensorwidth=5.0, xresolution=1280, yresolution=960):
    '''
    NOTE: this algorithm is incorrect
    
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
    
    pw = pixel_width(height, xresolution=xresolution, lens=lens, sensorwidth=sensorwidth)

    dx = (xresolution/2) - xpos
    dy = (yresolution/2) - ypos

    range_c = math.hypot(dx * pw, dy * py)
    angle_c = math.atan2(dy * pw, dx * px)

    # add in yaw
    angle_c += math.radians(yaw)

    # back to rectangular coordinates
    x = - range_c * math.cos(angle_c)
    y = range_c * math.sin(angle_c)

    return (xcenter+x, ycenter+y)

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
    from numpy import array,eye
    from cuav.uav.uav import uavxfer
    from math import pi
  
    xfer = uavxfer()
    xfer.setCameraMatrix(C.K)
    xfer.setCameraOrientation( 0.0, 0.0, pi/2 )
    xfer.setFlatEarth(0);
    xfer.setPlatformPose(0, 0, -height, math.radians(roll), math.radians(pitch), math.radians(yaw))

    # compute the undistorted points for the ideal camera matrix
    src = cv.CreateMat(1, 1, cv.CV_64FC2)
    src[0,0] = (xpos, ypos)
    dst = cv.CreateMat(1, 1, cv.CV_64FC2)
    R = cv.fromarray(eye(3))
    K = cv.fromarray(C.K)
    D = cv.fromarray(C.D)
    cv.UndistortPoints(src, dst, K, D, R, K)
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


def pixel_position(xpos, ypos, height, pitch, roll, yaw, C):
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
    from rotmat import Vector3, Matrix3, Plane, Line
    from math import radians
    
    # get pixel sizes in meters, this assumes we are pointing straight down with square pixels
    pw = pixel_width(height, xresolution=C.xresolution, lens=C.lens, sensorwidth=C.sensorwidth)

    # ground plane
    ground_plane = Plane()

    # the position of the camera in the air, remembering its a right
    # hand coordinate system, so +ve z is down
    camera_point = Vector3(0, 0, -height)

    # get position on ground relative to camera assuming camera is pointing straight down
    ground_point = Vector3(-pw * (ypos - (C.yresolution/2)),
			   pw * (xpos - (C.xresolution/2)),
			   height)
    
    # form a rotation matrix from our current attitude
    r = Matrix3()
    r.from_euler(radians(roll), radians(pitch), radians(yaw))

    # rotate ground_point to form vector in ground frame
    rot_point = r * ground_point

    # a line from the camera to the ground
    line = Line(camera_point, rot_point)

    # find the intersection with the ground
    pt = line.plane_intersection(ground_plane, forward_only=True)
    if pt is None:
	    # its pointing up into the sky
	    return None
    return (pt.y, pt.x)


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

def gps_position_from_xy(x, y, pos, C=CameraParams(), altitude=None):
    '''
    return a GPS position in an image given a MavPosition object
    and an image x,y position
    '''
    if pos is None:
            return None
    width=C.xresolution
    height=C.yresolution
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
        

def gps_position_from_image_region(region, pos, width=1280, height=960, C=CameraParams(), altitude=None):
    '''
    return a GPS position in an image given a MavPosition object
    and an image region tuple
    '''
    if pos is None:
        return None
    x = (region.x1+region.x2)*0.5
    y = (region.y1+region.y2)*0.5
    return gps_position_from_xy(x, y, pos, C=C, altitude=altitude)

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

def parse_frame_time(filename):
	'''parse a image frame time from a image filename
	from the chameleon capture code'''
	basename = os.path.basename(filename)
	i = basename.find('201')
	if i == -1:
                if basename.lower().endswith('.jpg') or basename.lower().endswith('.jpeg'):
                        from . import mav_position
                        return mav_position.exif_timestamp(filename)
		raise RuntimeError('unable to parse filename %s into time' % basename)
	tstring = basename[i:]
        ttuple = time.strptime(tstring[:14], "%Y%m%d%H%M%S")
	t = time.mktime(ttuple)
        dst = time.localtime().tm_isdst
	# hundredths can be after a dash
	if tstring[14] == '-':
		hundredths = int(tstring[15:17])
                z = tstring[17]
	else:
		hundredths = int(tstring[14:16])
                z = tstring[16]
	t += hundredths * 0.01
        if z.upper() == 'Z':
                if dst > 0:
                        t -= time.altzone
                else:
                        t -= time.timezone
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


		

def socket_send_queue_size(sock):
    '''return size of the TCP send queue for a socket'''
    import fcntl, termios, struct
    buf = struct.pack('@l', 0)
    ret = fcntl.ioctl(sock.fileno(), termios.TIOCOUTQ, buf)
    v, = struct.unpack('@l', ret)
    return v


def LoadImage(filename):
	'''wrapper around cv.LoadImage that also handles PGM.
	It always returns a colour image of the same size'''
	if filename.endswith('.pgm'):
		from ..image import scanner
		pgm = PGM(filename)
		im_full = numpy.zeros((960,1280,3),dtype='uint8')
		scanner.debayer(pgm.array, im_full)
		return cv.GetImage(cv.fromarray(im_full))
	return cv.LoadImage(filename)


class PickleStreamIn:
	'''a non-blocking pickle abstraction'''
	def __init__(self):
		self.objs = []
		self.io = ""
		self.prefix = ""
		self.bytes_needed = -1

	def write(self, buf):
		'''add some data from the stream'''
		while len(buf) != 0:
			if len(self.prefix) < 4:
				n = min(4 - len(self.prefix), len(buf))
				self.prefix += buf[:n]
				buf = buf[n:]
			if self.bytes_needed == -1 and len(self.prefix) == 4:
				(self.bytes_needed,) = struct.unpack('<I', self.prefix)
			n = min(len(buf), self.bytes_needed - len(self.io))
			self.io += buf[:n]
			buf = buf[n:]
			if len(self.io) == self.bytes_needed:
				self.objs.append(cPickle.loads(self.io))
				self.io = ""
				self.prefix = ""
				self.bytes_needed = -1

	def get(self):
		'''get an object if available'''
		if len(self.objs) == 0:
			return None
		return self.objs.pop(0)

class PickleStreamOut:
	'''a non-blocking pickle abstraction - output side'''
	def __init__(self, sock):
		self.sock = sock

	def send(self, obj):
		'''send an object over the stream'''
		buf = cPickle.dumps(obj, protocol=cPickle.HIGHEST_PROTOCOL)
		prefix = struct.pack('<I', len(buf))
		self.sock.send(prefix + buf)
		

def image_shape(img):
	'''return (w,h) of an image, coping with different image formats'''
	if getattr(img, 'shape', None) is not None:
		return (img.shape[1], img.shape[0])
	return (getattr(img, 'width'), getattr(img, 'height'))

def image_width(img):
	'''return width of an image, coping with different image formats'''
	if getattr(img, 'shape', None) is not None:
		return img.shape[1]
	return getattr(img, 'width')


def cv_wait_quit():
	'''wait until q is hit for quit'''
	print("Press q to quit")
	while True:
		key = cv.WaitKey()
		key &= 0xFF
		if not key in range(128):
			continue
		key = chr(key)
		if key in ['q', 'Q']:
			break

def SubImage(img, region):
	'''return a subimage as a new image. This allows
	for the region going past the edges.
	region is of the form (x1,y1,width,height)'''
	(x1,y1,width,height) = region
	zeros = numpy.zeros((height,width,3),dtype='uint8')
	ret = cv.GetImage(cv.fromarray(zeros))
	(img_width,img_height) = cv.GetSize(img)
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
	cv.SetImageROI(img, (sx1, sy1, w-xofs, h-yofs))
	cv.SetImageROI(ret, (xofs, yofs, w-xofs, h-yofs))
	cv.Copy(img, ret)
	cv.ResetImageROI(img)
	cv.ResetImageROI(ret)
	return ret

def OverlayImage(img, img2, x, y):
	'''overlay a 2nd image on a first image, at position x,y
	on the first image'''
	(w,h) = cv.GetSize(img2)
	cv.SetImageROI(img, (x, y, w, h))
	cv.Copy(img2, img)
	cv.ResetImageROI(img)

if __name__ == "__main__":
	pos1 = (-35.36048084339494,  149.1647973335984)
	pos2 = (-35.36594385616202,  149.1656758119368)
	dist = gps_distance(pos1[0], pos1[1],
			    pos2[0], pos2[1])
	bearing = gps_bearing(pos1[0], pos1[1],
			      pos2[0], pos2[1])
	print 'distance %f m' % dist
	print 'bearing %f degrees' % bearing
	pos3 = gps_newpos(pos1[0], pos1[1],
			  bearing, dist)
	err = gps_distance(pos2[0], pos2[1],
			   pos3[0], pos3[1])
	if math.fabs(err) > 0.01:
		raise RuntimeError('coordinate error too large %f' % err)
	# check negative distances too
	pos4 = gps_newpos(pos3[0], pos3[1],
			  bearing, -dist)
	err = gps_distance(pos1[0], pos1[1],
			   pos4[0], pos4[1])
	if math.fabs(err) > 0.01:
		raise RuntimeError('coordinate error too large %f' % err)
	print 'error %f m' % err

	print('Testing polygon_outside()')
	'''
	this is the boundary of the 2010 outback challenge
	Note that the last point must be the same as the first for the
	polygon_outside() algorithm
	'''
	
	OBC_boundary = polygon_load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'OBC_boundary.txt'))
	test_points = [
		(-26.6398870, 151.8220000, True ),
		(-26.6418700, 151.8709260, False ),
		(-350000000, 1490000000, True ),
		(0, 0,                   True ),
		(-26.5768150, 151.8408250, False ),
		(-26.5774060, 151.8405860, True ),
		(-26.6435630, 151.8303440, True ),
		(-26.6435650, 151.8313540, False ),
		(-26.6435690, 151.8303530, False ),
		(-26.6435690, 151.8303490, True ),
		(-26.5875990, 151.8344049, True ),
		(-26.6454781, 151.8820530, True ),
		(-26.6454779, 151.8820530, True ),
		(-26.6092109, 151.8747420, True ),
		(-26.6092111, 151.8747420, False ),
		(-26.6092110, 151.8747421, True ),
		(-26.6092110, 151.8747419, False ),
		(-26.6092111, 151.8747421, True ),
		(-26.6092109, 151.8747421, True ),
		(-26.6092111, 151.8747419, False ),
		(-27.6092111, 151.8747419, True ),
		(-27.6092111, 152.0000000, True ),
		(-25.0000000, 150.0000000, True )
		]
	if not polygon_complete(OBC_boundary):
		raise RuntimeError('OBC_boundary invalid')
	for lat, lon, outside in test_points:
		if outside != polygon_outside((lat, lon), OBC_boundary):
			raise RuntimeError('OBC_boundary test error', lat, lon)
			

def SaturateImage(img, scale=2, brightness=2):
	'''return a zoomed saturated image. Assumes a RGB image'''
	(w,h) = cv.GetSize(img)
	(w2,h2) = (w//scale, h//scale)
        cv.SetImageROI(img, ((w//2)-(w2//2),(h//2)-(h2//2),w2,h2))
        img2 = cv.CreateImage((w,h), 8, 3)
        cv.Resize(img, img2)
        cv.ResetImageROI(img)
	cv.CvtColor(img2, img2, cv.CV_RGB2HSV)	
        for x in range(w):
            for y in range(h):
                (hue,s,v) = img2[y,x]
                img2[y,x] = (hue,255,v)
        cv.CvtColor(img2, img2, cv.CV_HSV2RGB)
	if brightness != 1:
		cv.ConvertScale(img2, img2, scale=brightness)
	return img2



def zero_image(img):
	'''zero an image'''
        cv.ConvertScale(img, img, scale=0)


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
