#!/usr/bin/python
'''
display a set of found regions as a mosaic
Andrew Tridgell
May 2012
'''

import numpy, os, cv, sys, cuav_util

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
import scanner

class Mosaic():
  '''keep a mosaic of found regions'''
  def __init__(self, width=512, height=512):
    self.width = width
    self.height = height
    self.mosaic = numpy.zeros((width,height,3),dtype='uint8')
    self.num_regions = (width/32)*(height/32)
    self.region_index = 0
    self.regions = [None] * self.num_regions
    self.full_res = False
    cv.NamedWindow('Mosaic')

  def mouse_event(self, event, x, y, flags, data):
    '''called on mouse events'''
    if flags & cv.CV_EVENT_FLAG_RBUTTON:
      self.full_res = not self.full_res
    if not (flags & cv.CV_EVENT_FLAG_LBUTTON):
      return
    idx = (x/32) + (self.width/32)*(y/32)
    if self.regions[idx] is None:
      return
    (r, filename) = self.regions[idx]
    print '-> %s' % filename
    if filename.endswith('.pgm'):
      pgm = cuav_util.PGM(filename)
      im = numpy.zeros((pgm.array.shape[0],pgm.array.shape[1],3),dtype='uint8')
      scanner.debayer_full(pgm.array, im)
      mat = cv.fromarray(im)
    else:
      mat = cv.LoadImage(filename)
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
    

  def add_regions(self, regions, img, filename):
    '''add some regions'''
    for r in regions:
      (x1,y1,x2,y2) = r
      dest_x = (self.region_index * 32) % self.height
      dest_y = ((self.region_index * 32) / self.width) * 32
      midx = (x1+x2)/2
      midy = (y1+y2)/2
      for x in range(-16, 16):
        for y in range(-16, 16):
          if (y+midy < 0 or x+midx < 0 or
              y+midy >= img.shape[0] or x+midx >= img.shape[1]):
            px = 0
          else:
            px = img[y+midy, x+midx]
          self.mosaic[dest_y+y+16, dest_x+x+16] = px
      self.regions[self.region_index] = (r, filename)
      self.region_index += 1
      if self.region_index >= self.num_regions:
        self.region_index = 0
    mat = cv.fromarray(self.mosaic)
    cv.ShowImage('Mosaic', mat)
    cv.SetMouseCallback('Mosaic', self.mouse_event, self)
