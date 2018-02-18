from numpy import pi, sqrt, sin, cos, tan

#
# Based on "Redfearn's" formula from the The GDA Technical Manual
# http://www.icsm.gov.au/gda/gdatm/
#

# some ellipsoids
#        Semi major axis  Inverse flattening  Central Scale factor
# GRS80  6378137.0        298.257222101       0.9996
# WGS84  6378137.0        298.257223563       0.9996


#
class geodetic:
  CSF = 0.9996; #central scale factor
  FalseEasting = 500000.0;
  FalseNorthing = 10000000.0; # for southern hemisphere
  ZoneWidth = 6.0   #in degrees
  # Longitude of initial central meridian (Zone one)
  CMZ1 = -177.0;

  BAND_LOOKUP = 'CDEFGHJKLMNPQRSTUVWXX'

  def __init__(self, a=6378137.0, f_i=298.257222101):
    # semi-major axis
    self.a = a
    # flattening
    self.f = 1.0/f_i;
    # eccentricity squared
    self.e2 = (2.0 - self.f) * self.f;

  def computeZoneAndBand(self, lat, lon):
    if (lat < -80.0):
      if (lon < 0.0):
        band = 'A'
      else:
        band = 'B'
    elif (lat > 84.0):
      if (lon < 0.0):
        band = 'Y'
      else:
        band = 'Z'
    else:
      zone = int((lon-self.CMZ1+self.ZoneWidth/2.0)/self.ZoneWidth) + 1;
      band = self.BAND_LOOKUP[int(lat+80)/8]

    return (zone, band)

  def geoToGrid(self, lat_d, lon_d, zone, band):
    lat = lat_d*pi/180.0;

    # Meridian distance
    e2 = self.e2
    e4 = e2*e2;
    e6 = e4*e2;
    A0 = 1-(e2/4.0)-(3.0*e4/64.0)-(5.0*e6/256.0);
    A2 = (3.0/8.0)*(e2+e4/4.0+15.0*e6/128.0);
    A4 = (15.0/256.0)*(e4+3.0*e6/4.0);
    A6 = 35.0*e6/3072.0;
    s = sin(lat);
    s2 = sin(2.0*lat);
    s4 = sin(4.0*lat);
    s6 = sin(6.0*lat);
    m = self.a*(A0*lat-A2*s2+A4*s4-A6*s6);

    # Radii of curvature.
    rho = self.a*(1-e2)/((1.0-e2*s*s)**(3.0/2.0));
    nu = self.a/sqrt(1-e2*s*s);
    psi = nu / rho;
    psi2 = psi*psi;
    psi3 = psi*psi2;
    psi4 = psi*psi3;

    # Geographical to Grid
    # longitude of central meridian of zone (degrees)
    self.LongCMZ = (zone - 1) * self.ZoneWidth + self.CMZ1;
    # the arc distance from central meridian (radians)
    w = (lon_d - self.LongCMZ)*pi/180.0;
    w2 = w*w;
    w3 = w*w2;
    w4 = w*w3;
    w5 = w*w4;
    w6 = w*w5;
    w7 = w*w6;
    w8 = w*w7;

    c = cos(lat);
    c3 = c*c*c;
    c5 = c*c*c3;
    c7 = c*c*c5;

    t = tan(lat);
    t2 = t*t;
    t4 = t2*t2;
    t6 = t2*t4;

    # Northing
    term1 = w2*c/2.0;
    term2 = w4*c3*(4.0*psi2+psi-t2)/24.0;
    term3 = w6*c5*(8.0*psi4*(11.0-24.0*t2)-28*psi3*(1-6.0*t2)+psi2*(1-32*t2)-psi*(2.0*t2)+t4)/720.0;
    term4 = w8*c7*(1385.0-3111.0*t2+543.0*t4-t6)/40320.0;
    northing = self.CSF*(m+nu*s*(term1+term2+term3+term4));
    if (band < 'N'):
      northing += self.FalseNorthing

    # Easting
    term1 = w*c;
    term2 = w3*c3*(psi-t2)/6.0;
    term3 = w5*c5*(4.0*psi3*(1.0-6.0*t2)+psi2*(1.0+8.0*t2)-psi*(2.0*t2)+t4)/120.0;
    term4 = w7*c7*(61.0-479.0*t2+179.0*t4-t6)/5040.0;
    easting = nu*self.CSF*(term1+term2+term3+term4) + self.FalseEasting;

    return (northing, easting)

  def gridToGeo(self, northing, easting, zone, band):
    E_ = easting - self.FalseEasting
    N_ = northing;
    if (band < 'N'):
      N_ -= self.FalseNorthing
    m = N_/self.CSF
 
    # Foot-point Latitude
    n = self.f/(2.0-self.f)
    n2 = n*n
    n3 = n2*n
    n4 = n2*n2
    G = self.a*(1.-n)*(1.-n2)*(1.+(9./4.)*n2+(225./64.)*n4)*(pi/180.)

    sigma  = (m*pi)/(180*G)
    phi_ = sigma +\
      ((3.*n/2.)-(27.*n3/32.))*sin(2.*sigma) + \
      ((21.*n2/16.)-(55.*n4/32.))*sin(4.*sigma) +\
      (151.*n3/96.)*sin(6.*sigma) +\
      (1097.*n4/512.)*sin(8.*sigma);

    # Radii of curvature. (using foot point latitude)
    s_ = sin(phi_)
    e2 = self.e2
    rho_ = self.a*(1-e2)/((1.0-e2*s_*s_)**(3.0/2.0))
    nu_ = self.a/sqrt(1-e2*s_*s_)
    psi_ = nu_ / rho_
    psi2_ = psi_*psi_
    psi3_ = psi2_*psi_
    psi4_ = psi2_*psi2_

    x = E_/(self.CSF*nu_)
    x3 = x*x*x
    x5 = x3*x*x
    x7 = x5*x*x
    t_ = tan(phi_)
    t2_ = t_*t_;
    t4_ = t2_*t2_;
    t6_ = t2_*t4_;
    tkr_ = (t_/(self.CSF*rho_))
    term1 = tkr_*(x*E_/2.)
    term2 = tkr_*(E_*x3/24.)*(-4.*psi2_ + 9.*psi_*(1.-t2_) + 12.*t2_)
    term3 = tkr_*(E_*x5/720.)*(8.*psi4_*(11.-24.*t2_)\
              - 12.*psi3_*(21.-71.*t2_)\
              + 15.*psi2_*(15.-98.*t2_+15.*t4_)\
              + 180.*psi_*(5.*t2_-3.*t4_)\
              + 360.*t4_)
    term4 = tkr_*(E_*x7/40320.)*(1385. + 3633.*t2_ + 4095.*t4_ + 1575.*t6_)
    phi = phi_ - term1 +term2 - term3 + term4
    lat = phi*180/pi

    sec_phi_ = 1.0/cos(phi_)
    term1 = x * sec_phi_
    term2 = (x3/6.)*sec_phi_*(psi_ + 2.*t2_)
    term3 = (x5/120.)*sec_phi_*(-4.*psi3_*(1.0-6*t2_) + psi2_*(9.-68.*t2_)\
                                + 72.*psi_*t2_ + 24.*t4_)
    term4 = (x7/5040.)*sec_phi_*(61. + 662.*t2_ + 1320.*t4_ + 720.*t6_)
    w = term1 - term2 + term3 - term4
    lambda0 = (self.CMZ1+(zone-1)*self.ZoneWidth)*pi/180.0;

    _lambda = lambda0 + w
    lon = _lambda*180.0/pi
    return (lat, lon)

