from numpy import array, linalg, eye, zeros, dot, transpose
from numpy import sin, cos, pi

#def rotationMatrix(phi, theta, psi):
#  out = zeros((3,3))
#  out[0,0] = cos(psi)*cos(theta)
#  out[0,1] = cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)
#  out[0,2] = cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)

#  out[1,0] = sin(psi)*cos(theta)
#  out[1,1] = sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi)
#  out[1,2] = sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)

#  out[2,0] = -sin(theta)
#  out[2,1] = cos(theta)*sin(phi)
#  out[2,2] = cos(theta)*cos(phi)

#  return out

# a simple frame transformation matrix
def rotationMatrix(phi, theta, psi):
  R_phi   = array([[1.0,0.0,0.0],[0.0,cos(phi),sin(phi)],[0.0,-sin(phi),cos(phi)]])
  R_theta = array([[ cos(theta),0.0,-sin(theta)],[0.0,1.0,0.0],[ sin(theta),0.0,cos(theta)]])
  R_psi   = array([[cos(psi),sin(psi),0.0],[-sin(psi),cos(psi),0.0],[0.0,0.0,1.0]])
  R = dot(dot(R_phi,R_theta),R_psi)
  return R


class uavxfer:
  
  def setCameraParams(self, fu, fv, cu, cv):
    K  = array([[fu, 0.0, cu],[0.0, fv, cv],[0.0, 0.0, 1.0]])
    self.setCameraMatrix(K)

  def setCameraMatrix(self, K):
    K_i = linalg.inv(K)
    self.Tk = eye(4,4)
    self.Tk[:3,:3] = K;
    self.Tk_i = eye(4,4)
    self.Tk_i[:3,:3] = K_i

  def setCameraOrientation(self, roll, pitch, yaw):
    self.Rc = array(eye(4,4))
    self.Rc[:3,:3] = transpose(rotationMatrix(roll, pitch, yaw))
    self.Rc_i = linalg.inv(self.Rc)

  def setPlatformPose(self, north, east, down, roll, pitch, yaw):
    self.Xp = array([north, east, down, 1.0])
    self.Rp = array(eye(4,4))
    self.Rp[:3,:3] = transpose(rotationMatrix(roll, pitch, yaw))
    self.Rp[:3,3] = array([north, east, down])
    self.Rp_i = linalg.inv(self.Rp)

  def setFlatEarth(self, z):
    self.z_earth = z

  def worldToPlatform(self, north, east, down):
    x_w = array([north, east, down, 1.0])
    x_p = dot(self.Rp_i, x_w)[:3]
    return x_p

  def worldToImage(self, north, east, down):
    x_w = array([north, east, down, 1.0])
    x_p = dot(self.Rp_i, x_w)
    x_c = dot(self.Rc_i, x_p)
    x_i = dot(self.Tk, x_c)
    return x_i[:3]/x_i[2]

  def platformToWorld(self, north, east, down):
    x_p = array([north, east, down, 1.0])
    x_w = dot(self.Rp, x_p)
    return x_w

  def imageToWorld(self, u, v):
    x_i = array([u, v, 1.0, 0.0])
    v_c = dot(self.Tk_i, x_i)
    v_p = dot(self.Rc, v_c)
    v_w = dot(self.Rp, v_p)
    # compute scale for z == z_earth
    scale = (self.z_earth-self.Xp[2])/v_w[2]
    #project from platform to ground
    x_w = scale*v_w + self.Xp;
    return x_w, scale

  def __init__(self, fu=200, fv=200, cu=512, cv=480):
    self.setCameraParams(fu, fv, cu, cv)
    self.Rc = self.Rc_i = array(eye(4,4))
    self.Rp = self.Rp_i = array(eye(4,4))
    self.z_earth = -600


