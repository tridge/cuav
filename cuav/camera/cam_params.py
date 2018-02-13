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
import sys

class CameraParams:
  # A default constructor based on sensor and lens specs only
  def __init__(self, lens=None, sensorwidth=None, xresolution=None, yresolution=None, K=None, D=None):
    if lens is None:
      raise ValueError("Lens required")
    if sensorwidth is None:
      raise ValueError("sensorwidth required")
    if xresolution is None:
      raise ValueError("yresolution required")
    if yresolution is None:
      raise ValueError("yresolution required")
    self.version = 0
    self.sensorwidth = sensorwidth
    self.lens = lens
    self.K = K
    self.D = D
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
    if self.K is not None:
      data['K'] = self.K.tolist()
    if self.D is not None:
      data['D'] = self.D.tolist()
    return data

  @staticmethod
  def fromdict(data):
    if data['version'] == 0:
      try:
        K = array(data['K'])
        D = array(data['D'])
      except KeyError:
        K = None
        D = None
      ret = CameraParams(lens=data['lens'],
                         sensorwidth=data['sensorwidth'],
                         xresolution=data['xresolution'],
                         yresolution=data['yresolution'],
                         K=K,
                         D=D)
      ret.version = data['version']
      return ret;
    else:
      raise Exception('version %d of camera params unsupported' % (self.version))

  @staticmethod
  def fromstring(strung):
    dic = json.loads(strung)
    return CameraParams.fromdict(dic)

  def save(self, filename):
    f = open(filename,"wb")
    # dump json form
    f.write(str(self)+"\n")
    f.close()

  @staticmethod
  def fromfile(filename):
    f = open(filename,"rb")
    # dump json form
    d = f.read(65535)
    f.close()
    return CameraParams.fromstring(d)

if __name__ == "__main__":
  import json
  C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1280, yresolution=960)
  C.save('foo.txt')
  print(C)
  C2 = CameraParams.fromfile('foo.txt')
  print(C2)
  if str(C) != str(C2):
    print("Reload mismatch")
    sys.exit(1)
  sys.exit(0)
