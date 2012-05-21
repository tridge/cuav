#!/usr/bin/python
'''
display a set of found regions as a mosaic
Andrew Tridgell
May 2012
'''

import numpy, os, cv, sys, cuav_util, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
import scanner

class Mosaic():
  '''keep a mosaic of found regions'''
  def __init__(self, width=512, height=512):
    self.width = width
    self.height = height
    self.thumb_size = 32
    self.mosaic = numpy.zeros((width,height,3),dtype='uint8')
    self.num_regions = (width/self.thumb_size)*(height/self.thumb_size)
    self.region_index = 0
    self.regions = [None] * self.num_regions
    self.full_res = False
    cv.NamedWindow('Mosaic')

  def mouse_event(self, event, x, y, flags, data):
    '''called on mouse events'''
    if flags & cv.CV_EVENT_FLAG_RBUTTON:
      self.full_res = not self.full_res
      print("full_res: %s" % self.full_res)
    if not (flags & cv.CV_EVENT_FLAG_LBUTTON):
      return
    idx = (x/self.thumb_size) + (self.width/self.thumb_size)*(y/self.thumb_size)
    if self.regions[idx] is None:
      return
    (r, filename, pos) = self.regions[idx]
    if pos:
      (lat, lon) = cuav_util.gps_position_from_image_region(r, pos)
      position_string = '%f %f %.1f %s %s' % (lat, lon, pos.altitude, pos, time.asctime(time.localtime(pos.time)))
    else:
      position_string = ''
    print '-> %s %s' % (filename, position_string)
    if filename.endswith('.pgm'):
      pgm = cuav_util.PGM(filename)
      im = numpy.zeros((pgm.array.shape[0],pgm.array.shape[1],3),dtype='uint8')
      scanner.debayer_full(pgm.array, im)
      mat = cv.fromarray(im)
    else:
      mat = cv.LoadImage(filename)
      im = numpy.asarray(cv.GetMat(mat))
    (x1,y1,x2,y2) = r
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

    cv.ShowImage('Mosaic Image', mat)
    cv.WaitKey(1)
    

  def add_regions(self, regions, img, filename, pos=None):
    '''add some regions'''
    if getattr(img, 'shape', None) is None:
      img = numpy.asarray(cv.GetMat(img))
    for r in regions:
      (x1,y1,x2,y2) = r

      midx = (x1+x2)/2
      midy = (y1+y2)/2
      x1 = midx - 15
      y1 = midy - 15
      if x1 < 0: x1 = 0
      if y1 < 0: y1 = 0
      
      # leave a 1 pixel black border
      thumbnail = numpy.zeros((self.thumb_size-1,self.thumb_size-1,3),dtype='uint8')
      scanner.rect_extract(img, thumbnail, x1, y1)

      dest_x = (self.region_index * self.thumb_size) % self.width
      dest_y = ((self.region_index * self.thumb_size) / self.width) * self.thumb_size

      # overlay thumbnail on mosaic
      scanner.rect_overlay(self.mosaic, thumbnail, dest_x, dest_y)

      self.regions[self.region_index] = (r, filename, pos, thumbnail)
      self.region_index += 1
      if self.region_index >= self.num_regions:
        self.region_index = 0
    mat = cv.fromarray(self.mosaic)
    cv.ShowImage('Mosaic', mat)
    cv.SetMouseCallback('Mosaic', self.mouse_event, self)
