from numpy import array, linalg, eye, zeros, dot
from numpy import sin, cos, pi

def rotationMatrix(phi, theta, psi):
  out = zeros((3,3))
  out[0,0] = cos(psi)*cos(theta)
  out[0,1] = cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)
  out[0,2] = cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)

  out[1,0] = sin(psi)*cos(theta)
  out[1,1] = sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi)
  out[1,2] = sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)

  out[2,0] = -sin(theta)
  out[2,1] = cos(theta)*sin(phi)
  out[2,2] = cos(theta)*cos(phi)

  return out


class uavxfer:
  
  def setCameraParams(self, fu, fv, cu, cv):
    self.K  = array([[fu, 0.0, cu, 0.0],[0.0, fv, cv, 0.0],[0.0, 0.0, 1.0, 0.0]])
    self.K_i = linalg.pinv(self.K)

  def setCameraOrientation(self, roll, pitch, yaw):
    self.Rc = array(eye(4,4))
    self.Rc[:3,:3] = rotationMatrix(roll, pitch, yaw)
    self.Rc_i = linalg.inv(self.Rc)

  def setPlatformPose(self, north, east, down, roll, pitch, yaw):
    self.Rp = array(eye(4,4))
    self.Rp[:3,:3] = rotationMatrix(roll, pitch, yaw)
    self.Rp[:3,3] = array([north, east, down])
    self.Rp_i = linalg.inv(self.Rp)

  def setFlatEarth(self, z):
    self.z_earth = z

  def worldToPlatform(self, north, east, down):
    x_w = array([north, east, down, 1.0])
    return dot(self.Rp_i, x_w)[:3]

  def worldToImage(self, north, east, down):
    x_w = array([north, east, down, 1.0])
    x_p = dot(self.Rp_i, x_w)
    x_i = dot(self.K, x_p)
    return x_i[:3]/x_i[2]

  def platformToWorld(self, north, east, down):
    return array([0.0, 0.0, 0.0])

  def imageToWorld(self, u, v):
    return array([0.0, 0.0, 0.0])

  def __init__(self, fu, fv, cu, cv):
    self.setCameraParams(fu, fv, cu, cv)
    self.Rc = self.Rc_i = array(eye(4,4))
    self.Rp = self.Rp_i = array(eye(4,4))
    self.z_earth = 0


if __name__ == '__main__':
  xfer = uavxfer(200.0, 200.0, 512, 480)
  xfer.setCameraOrientation(0.0,0.0,0.0)
  xfer.setPlatformPose(500.0, 1000.0, -700.0, 0.00, 0.0, 0.0)

  p_w = array([500, 1000.0, 0.0])
  print xfer.worldToPlatform(p_w[0], p_w[1], p_w[2])
  print xfer.worldToImage(p_w[0], p_w[1], p_w[2])



  print('foo')
