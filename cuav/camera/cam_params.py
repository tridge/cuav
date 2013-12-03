#!/usr/bin/env python
#
# Camera class to store/save/load camera/lens parameters
#
# Matthew Ridley, July 2012
#

'''Camera params
'''

from numpy import array
import json
from exceptions import Exception

class CameraParams:
  # A default constructor based on sensor and lens specs only
  def __init__(self, lens=4.0, sensorwidth=5.0, xresolution=1280, yresolution=960):
    self.version = 0
    self.sensorwidth = sensorwidth
    self.lens = lens
    self.set_resolution(xresolution, yresolution)

  def set_resolution(self, xresolution, yresolution):
    '''set camera resolution'''
    self.xresolution = xresolution
    self.yresolution = yresolution

    # compute focal length in pixels
    f_p = xresolution * self.lens / self.sensorwidth

    self.K = array([[f_p, 0.0, xresolution/2],[0.0, f_p, yresolution/2], [0.0,0.0,1.0]])
    self.D = array([[0.0, 0.0, 0.0, 0.0, 0.0]])

  def __repr__(self):
      return json.dumps(self.todict(),indent=2)

  def setParams(self, K, D):
    self.K = array(K)
    self.D = array(D)

  def todict(self):
    data = {}
    data['version'] = self.version
    data['lens'] = self.lens
    data['sensorwidth'] = self.sensorwidth
    data['xresolution'] = self.xresolution
    data['yresolution'] = self.yresolution
    data['K'] = self.K.tolist()
    data['D'] = self.D.tolist()
    return data

  def fromdict(self, data):
    self.version = data['version']
    if self.version == 0:
      self.lens = data['lens']
      self.sensorwidth = data['sensorwidth']
      self.xresolution = data['xresolution']
      self.yresolution = data['yresolution']
      self.K = array(data['K'])
      self.D = array(data['D'])
    else:
      raise Exception('version %d of camera params unsupported' % (self.version))

  def fromstring(self, strung):
    dic = json.loads(strung)
    self.fromdict(dic)

  def save(self, filename):
    f = open(filename,"wb")
    # dump json form
    f.write(str(self)+"\n")
    f.close()

  def load(self, filename):
    f = open(filename,"rb")
    # dump json form
    d = f.read(65535)
    f.close()
    self.fromstring(d)

if __name__ == "__main__":
  import json
  C = CameraParams()
  C.save('foo.txt')
  print C
  C.load('foo.txt')
  print C
