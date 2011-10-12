from uav import uavxfer
from geo import geodetic
from matplotlib import pyplot

if __name__ == '__main__':
  from optparse import OptionParser
  from numpy import linalg,pi,mean,cov,sqrt,loadtxt,arange

  inputfile = '../../../data/2011-05-01/joeposyaw.txt';

  data = loadtxt(inputfile, usecols=arange(1,9))

  files = loadtxt(inputfile, dtype='string', usecols=(0,))

  #parser = OptionParser("showjoe.py [options]")

  #(opts, args) = parser.parse_args()
  f_m = 4  # 4mm
  #s_m = (4./5.)*25.4/3 #1/3" sensor width in mm
  s_m = 5.05

  s_p = 1280
  f_p = s_p * f_m / s_m
  par = 1.0
  f_u = f_p*par
  f_v = f_p/par

  xfer = uavxfer()
  xfer.setCameraParams(f_u, f_v, 640, 480)
  xfer.setCameraOrientation( -0.1, 0.1, pi/2 )
  height0 = 0.0
  xfer.setFlatEarth(-height0);
  g = geodetic()
  lat0 = data[0,2]
  lon0 = data[0,3]

  (zone, band) = g.computeZoneAndBand(lat0, lon0)
  (north0, east0) = g.geoToGrid(lat0, lon0, zone, band)
  print 'zone/band:',zone,band

  f = pyplot.figure(1)
  f.clf()

  joes = [];

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
    if (abs(roll) < 33 and abs(pitch) < 33 and scale > 0 and scale < 500 ):
      #print 'platform: north =',north, 'east =', east
      #print '    roll: ', roll, 'pitch:', pitch, ' yaw:', yaw
      #print '   image: ', im_x, im_y
      #print '     joe: north =', north0+joe_w[0], 'east =', east+joe_w[1], 'scale =', scale
      pyplot.plot(joe_w[1], joe_w[0], 'bx')
      pyplot.plot(pe, pn, 'ro')
      joes.insert(0, [joe_w[1], joe_w[0]]);
  pyplot.axis('equal')
  f.show()

  joe = mean(joes, 0)
  joe_e = joe[0];
  joe_n = joe[1];
  (joe_lat, joe_lon) = g.gridToGeo(north0+joe_n, east0+joe_e, zone, band)

  print 'joe is located @', joe_lat, joe_lon
  joe_cov = cov(joes, rowvar=0)
  print 'covariance of joe\n', joe_cov


