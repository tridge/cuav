from uav import uavxfer
from geo import geodetic

if __name__ == '__main__':
  import numpy
  from optparse import OptionParser

  inputfile = '../../../data/2011-05-01/joeposyaw.txt';

  data = numpy.loadtxt(inputfile, usecols=numpy.arange(1,9))

  files = numpy.loadtxt(inputfile, dtype='string', usecols=(0,))

  #parser = OptionParser("showjoe.py [options]")
  #(opts, args) = parser.parse_args()

  for i in range(0, data.shape[0]):
    im_x = data[i,0]
    im_y = data[i,1]
    lat = data[i,2]
    lon = data[i,3]
    height = data[i,4]
    yaw = data[i,5]
    pitch = data[i,6]
    roll = data[i,7]


