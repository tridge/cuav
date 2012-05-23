#!/usr/bin/python
'''
display a set of found regions as a mosaic
Andrew Tridgell
May 2012
'''

import numpy, os, cv, cv2, sys, cuav_util, time, math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
import scanner

def compute_signatures(hist1, hist2):
    '''
    demos how to convert 2 histograms into 2 signature
    '''
    h_bins = hist1.bins.shape(0)
    s_bins = hist1.bins.shape(1)
    num_rows = h_bins * s_bins
    sig1 = cv.CreateMat(num_rows, 3, cv.CV_32FC1)
    sig2 = cv.CreateMat(num_rows, 3, cv.CV_32FC1)
    #fill signatures
    #TODO: for production optimize this, use Numpy
    for h in range(0, h_bins):
        for s in range(0, s_bins):
            bin_val = cv.QueryHistValue_2D(hist1, h, s)
            cv.Set2D(sig1, h*s_bins + s, 0, bin_val) #bin value
            cv.Set2D(sig1, h*s_bins + s, 1, h)  #coord1
            cv.Set2D(sig1, h*s_bins + s, 2, s) #coord2
            #signature.2
            bin_val2 = cv.QueryHistValue_2D(hist2, h, s)
            cv.Set2D(sig2, h*s_bins + s, 0, bin_val2) #bin value
            cv.Set2D(sig2, h*s_bins + s, 1, h)  #coord1
            cv.Set2D(sig2, h*s_bins + s, 2, s) #coord2

    return (sig1, sig2)

class MosaicRegion:
  def __init__(self, region, filename, pos, thumbnail, latlon=(None,None), map_pos=(None,None)):
    # self.region is a (minx,miny,maxy,maxy) rectange in image coordinates
    self.region = region
    self.filename = filename
    self.pos = pos
    self.thumbnail = thumbnail
    self.map_pos = map_pos
    self.latlon = latlon

  def __str__(self):
    if self.latlon != (None,None):
      position_string = '%s %.1f %s %s' % (
        self.latlon, self.pos.altitude, self.pos, time.asctime(time.localtime(self.pos.time)))
    else:
      position_string = ''
    return '%s %s' % (self.filename, position_string)
    

class Mosaic():
  '''keep a mosaic of found regions'''
  def __init__(self, grid_width=30, grid_height=30, thumb_size=20):
    self.thumb_size = thumb_size
    self.width = grid_width * thumb_size
    self.height = grid_height * thumb_size
    self.map_width = grid_width * thumb_size
    self.map_height = grid_height * thumb_size
    self.mosaic = numpy.zeros((self.height,self.width,3),dtype='uint8')
    self.map = numpy.zeros((self.map_height,self.map_width,3),dtype='uint8')
    self.map_background = numpy.zeros((self.map_height,self.map_width,3),dtype='uint8')
    self.display_regions = grid_width*grid_height
    self.regions = []
    self.full_res = False
    self.boundary = []
    self.fill_map = False

    # map limits in form (min_lat, min_lon, max_lat, max_lon)
    self.map_limits = [0,0,0,0]
    cv.NamedWindow('Mosaic')

  def latlon_to_map(self, lat, lon):
    '''
    convert a latitude/longitude to map position in pixels, with
    0,0 in top left
    '''
    dlat = lat - self.map_limits[0]
    dlon = lon - self.map_limits[1]
    px = (dlon / (self.map_limits[3] - self.map_limits[1])) * self.map_width
    py = (1.0 - dlat / (self.map_limits[2] - self.map_limits[0])) * self.map_height
    return (int(px+0.5), int(py+0.5))
  

  def set_boundary(self, boundary):
    '''set a polygon search boundary'''
    if not cuav_util.polygon_complete(boundary):
      raise RuntimeError('invalid boundary passed to mosaic')
    self.boundary = boundary[:]
    self.map_limits = [boundary[0][0], boundary[0][1],
                       boundary[0][0], boundary[0][1]]
    for b in self.boundary:
      (lat,lon) = b
      self.map_limits[0] = min(self.map_limits[0], lat)
      self.map_limits[1] = min(self.map_limits[1], lon)
      self.map_limits[2] = max(self.map_limits[2], lat)
      self.map_limits[3] = max(self.map_limits[3], lon)

    # add a 50m border
    (lat, lon) = cuav_util.gps_newpos(self.map_limits[0], self.map_limits[1],
                                      225, 50)
    self.map_limits[0] = lat
    self.map_limits[1] = lon

    (lat, lon) = cuav_util.gps_newpos(self.map_limits[2], self.map_limits[3],
                                      45, 50)
    self.map_limits[2] = lat
    self.map_limits[3] = lon

    # draw the border
    img = cv.GetImage(cv.fromarray(self.map))
    for i in range(len(self.boundary)-1):
      (lat1,lon1) = self.boundary[i]
      (lat2,lon2) = self.boundary[i+1]
      cv.Line(img,
              self.latlon_to_map(lat1, lon1),
              self.latlon_to_map(lat2, lon2),
              (0,255,0), 3)
    self.map = numpy.asarray(cv.GetMat(img))
    cv.ShowImage('Mosaic Map', cv.fromarray(self.map))
    cv.SetMouseCallback('Mosaic Map', self.mouse_event_map, self)


  def histogram(self, image, bins=10):
    '''return a 2D HS histogram of an image'''
    mat = cv.fromarray(image)
    hsv = cv.CreateImage(cv.GetSize(mat), 8, 3)
    cv.CvtColor(mat, hsv, cv.CV_BGR2Lab)
    planes = [ cv.CreateMat(mat.rows, mat.cols, cv.CV_8UC1),
               cv.CreateMat(mat.rows, mat.cols, cv.CV_8UC1) ]
    cv.Split(hsv, planes[0], planes[1], None, None)
    hist = cv.CreateHist([bins, bins], cv.CV_HIST_ARRAY)
    cv.CalcHist([cv.GetImage(p) for p in planes], hist)
    cv.NormalizeHist(hist, 1024)
    return hist

  def show_hist(self, hist):
    print(len(hist.bins))
    bins = len(hist.bins)
    a = numpy.zeros((bins, bins), dtype='uint16')
    for h in range(bins):
      for s in range(bins):
        a[h,s] = cv.QueryHistValue_2D(hist, h, s)
    print a
          

  def measure_distinctiveness(self, image, r):
    '''measure the distinctiveness of a region in an image'''
    if image.shape[0] == 960:
      r = tuple([2*v for v in r])
    (x1,y1,x2,y2) = r

    hist1 = self.histogram(image)
    self.show_hist(hist1)

    thumbnail = numpy.zeros((y2-y1+1, x2-x1+1,3),dtype='uint8')
    scanner.rect_extract(image, thumbnail, x1, y1)

    hist2 = self.histogram(thumbnail)
    self.show_hist(hist2)

    (sig1, sig2) = compute_signatures(hist1, hist2, h_bins=32, s_bins=32)
    return cv.CalcEMD2(sig1, sig2, cv.CV_DIST_L2, lower_bound=float('inf'))

    print cv.CompareHist(hist1, hist2, cv.CV_COMP_CHISQR)

  def display_region_image(self, region):
    '''display the image associated with a region'''
    jpeg_thumb = scanner.jpeg_compress(region.thumbnail, 75)
    print '-> %s thumb=%u' % (str(region), len(jpeg_thumb))
    if region.filename.endswith('.pgm'):
      pgm = cuav_util.PGM(region.filename)
      im = numpy.zeros((pgm.array.shape[0],pgm.array.shape[1],3),dtype='uint8')
      scanner.debayer_full(pgm.array, im)
      mat = cv.fromarray(im)
    else:
      mat = cv.LoadImage(region.filename)
      im = numpy.asarray(cv.GetMat(mat))
    (x1,y1,x2,y2) = region.region
    if im.shape[0] == 960:
      x1 *= 2
      y1 *= 2
      x2 *= 2
      y2 *= 2
    cv.Rectangle(mat, (x1-8,y1-8), (x2+8,y2+8), (255,0,0), 2)
    if not self.full_res:
      display_img = cv.CreateImage((640, 480), 8, 3)
      cv.Resize(mat, display_img)
      mat = display_img

    #print(self.measure_distinctiveness(im, r))

    cv.ShowImage('Mosaic Image', mat)
    cv.WaitKey(1)


  def mouse_event(self, event, x, y, flags, data):
    '''called on mouse events'''
    if flags & cv.CV_EVENT_FLAG_RBUTTON:
      self.full_res = not self.full_res
      print("full_res: %s" % self.full_res)
    if not (flags & cv.CV_EVENT_FLAG_LBUTTON):
      return
    # work out which region they want, taking into account wrap
    idx = (x/self.thumb_size) + (self.width/self.thumb_size)*(y/self.thumb_size)
    if idx >= len(self.regions):
      return
    first = (len(self.regions)/self.display_regions) * self.display_regions
    idx += first
    if idx >= len(self.regions):
      idx -= self.display_regions
    region = self.regions[idx]

    self.display_region_image(region)


  def find_closest_region(self, x, y):
    '''find the closest region given a pixel position on the map'''
    best_idx = 0
    best_distance = math.sqrt((self.regions[0].map_pos[0] - x)**2 + (self.regions[0].map_pos[1] - y)**2)
    for idx in range(len(self.regions)-1):
      distance = math.sqrt((self.regions[idx].map_pos[0] - x)**2 + (self.regions[idx].map_pos[1] - y)**2)
      if distance < best_distance:
        best_distance = distance
        best_idx = idx
    if best_distance > self.thumb_size:
      return None
    return self.regions[best_idx]

  def mouse_event_map(self, event, x, y, flags, data):
    '''called on mouse events on the map'''
    if flags & cv.CV_EVENT_FLAG_RBUTTON:
      self.full_res = not self.full_res
      print("full_res: %s" % self.full_res)
    if not (flags & cv.CV_EVENT_FLAG_LBUTTON):
      return

    region = self.find_closest_region(x,y)
    if region is None:
      return    
    self.display_region_image(region)

  def refresh_map(self):
    '''refresh the map display'''
    if self.fill_map:
      map = numpy.zeros((self.map_height,self.map_width,3),dtype='uint8')
      scanner.rect_overlay(map, self.map_background, 0, 0, False)
      scanner.rect_overlay(map, self.map, 0, 0, True)
    else:
      map = self.map
    cv.ShowImage('Mosaic Map', cv.fromarray(map))

  def add_image(self, img, pos):
    '''add a background image'''
    if not self.fill_map:
      return
    # show transformed image on map
    w = img.shape[1]
    h = img.shape[0]
    srcpos = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]
    dstpos = []
    for (x,y) in srcpos:
      (lat,lon) = cuav_util.pixel_coordinates(x, y, pos.lat, pos.lon, pos.altitude,
                                              pos.pitch, pos.roll, pos.yaw,
                                              xresolution=w, yresolution=h)
      dstpos.append(self.latlon_to_map(lat, lon))
    transform = cv2.getPerspectiveTransform(numpy.array(srcpos, dtype=numpy.float32), numpy.array(dstpos, dtype=numpy.float32))
    map_bg = cv.GetImage(cv.fromarray(self.map_background))
    cv.WarpPerspective(cv.fromarray(img), map_bg, cv.fromarray(transform), flags=0)
    self.map_background = numpy.asarray(cv.GetMat(map_bg))
    self.refresh_map()

  def add_regions(self, regions, img, filename, pos=None):
    '''add some regions'''
    if getattr(img, 'shape', None) is None:
      img = numpy.asarray(cv.GetMat(img))
    for r in regions:
      (x1,y1,x2,y2) = r

      (mapx, mapy) = (None, None)
      (lat, lon) = (None, None)

      if self.boundary and pos:
        (lat, lon) = cuav_util.gps_position_from_image_region(r, pos)
        if cuav_util.polygon_outside((lat, lon), self.boundary):
          # this region is outside the search boundary
          continue

      midx = (x1+x2)/2
      midy = (y1+y2)/2
      x1 = midx - self.thumb_size/2
      y1 = midy - self.thumb_size/2
      if x1 < 0: x1 = 0
      if y1 < 0: y1 = 0
      
      # leave a 1 pixel black border
      thumbnail = numpy.zeros((self.thumb_size-1,self.thumb_size-1,3),dtype='uint8')
      scanner.rect_extract(img, thumbnail, x1, y1)

      idx = len(self.regions) % self.display_regions
      
      dest_x = (idx * self.thumb_size) % self.width
      dest_y = ((idx * self.thumb_size) / self.width) * self.thumb_size

      # overlay thumbnail on mosaic
      scanner.rect_overlay(self.mosaic, thumbnail, dest_x, dest_y, False)

      if (lat,lon) != (None,None):
        # show thumbnail on map
        (mapx, mapy) = self.latlon_to_map(lat, lon)
        scanner.rect_overlay(self.map, thumbnail,
                             max(0, mapx - self.thumb_size/2),
                             max(0, mapy - self.thumb_size/2), False)
        map = cv.fromarray(self.map)
        (x1,y1) = (max(0, mapx - self.thumb_size/2),
                   max(0, mapy - self.thumb_size/2))
        (x2,y2) = (x1+self.thumb_size, y1+self.thumb_size)
        cv.Rectangle(map, (x1,y1), (x2,y2), (255,0,0), 1)
        self.map = numpy.asarray(map)
        self.refresh_map()

      self.regions.append(MosaicRegion(r, filename, pos, thumbnail, latlon=(lat, lon), map_pos=(mapx, mapy)))

    cv.ShowImage('Mosaic', cv.fromarray(self.mosaic))
    cv.SetMouseCallback('Mosaic', self.mouse_event, self)
