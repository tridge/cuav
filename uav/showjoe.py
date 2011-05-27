from uav import uavxfer

from geo import geodetic
from matplotlib import pyplot

if __name__ == '__main__':
  import numpy
  from optparse import OptionParser
  from numpy import linalg
  from numpy import pi

  inputfile = '../../../data/2011-05-01/joeposyaw.txt';

  data = numpy.loadtxt(inputfile, usecols=numpy.arange(1,9))

  files = numpy.loadtxt(inputfile, dtype='string', usecols=(0,))

  #parser = OptionParser("showjoe.py [options]")

  #(opts, args) = parser.parse_args()

  ff = 250
  par = 1.0
  fu = ff*par
  fv = ff/par

  xfer = uavxfer()
  xfer.setCameraParams(fu, fv, 640, 480)
  xfer.setCameraOrientation( -0.1, 0.1, pi/2 )
  height0 = data[0,4]
  xfer.setFlatEarth(-height0);
  g = geodetic()
  lat0 = data[0,2]
  lon0 = data[0,3]

  (zone, band) = g.computeZoneAndBand(lat0, lon0)
  (north0, east0) = g.geoToGrid(lat0, lon0, zone, band)
  print 'zone/band:',zone,band

  f = pyplot.figure(1)
  f.clf()

  for i in range(0, data.shape[0]):
    #for i in range(20,45):
    im_x = data[i,0]
    im_y = data[i,1]
    lat = data[i,2]
    lon = data[i,3]
    height = data[i,4]
    yaw = data[i,5]
    pitch = data[i,6]
    roll = data[i,7]
    (north, east) = g.geoToGrid(lat, lon, zone, band)
    down = -height
    pn = north-north0
    pe = east-east0
    xfer.setPlatformPose(pn, pe , down, pi*roll/180, pi*pitch/180, pi*yaw/180)
    #xfer.setPlatformPose(pn, pe , down, 0.0, 0.0, pi*yaw/180.0)
    (joe_w, scale) = xfer.imageToWorld(im_x, im_y)
    #(joe_w, scale) = xfer.imageToWorld(640.0, 380.0)

    # large roll or pitch are unreliable
    # negative scale means camera pointing above horizon
    # large scale means a long way away also unreliable
    if (abs(roll) < 33 and abs(pitch) < 33 and scale > 0 and scale < 150 ):
      print 'platform: north =',north, 'east =', east
      print '    roll: ', roll, 'pitch:', pitch, ' yaw:', yaw
      print '   image: ', im_x, im_y
      print '     joe: north =', north0+joe_w[0], 'east =', east+joe_w[1], 'scale =', scale
      pyplot.plot(joe_w[1], joe_w[0], 'bx')
      pyplot.plot(pe, pn, 'ro')
  pyplot.axis('equal')
  f.show()





