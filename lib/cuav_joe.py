#!/usr/bin/python
'''
object to hold Joe positions
We store a pickled list of these objects in joe.log
Andrew Tridgell
May 2012
'''

import os, sys, cuav_util, mav_position, cPickle, time


class JoePosition():
  '''a Joe position'''
  def __init__(self, latlon, frame_time, r,
               pos, image_filename):
      self.latlon = latlon
      self.frame_time = frame_time
      self.pos = pos
      self.r = r
      self.image_filename = image_filename

  def __str__(self):
    return 'JoePosition(lat=%f lon=%f %s %s %s %s raw%s.pgm)' % (self.latlon[0], self.latlon[1],
                                                                 self.pos, self.image_filename,
                                                                 str(getattr(self,'r','')),
                                                                 time.asctime(time.localtime(self.frame_time)),
                                                                 cuav_util.frame_time(self.frame_time))
      
class JoeLog():
  '''a Joe position logger'''
  def __init__(self, filename, append=True):
    self.filename = filename
    self.log = open(filename, "w+" if append else "w")
        
  def add(self, latlon, frame_time, r, pos, image_filename):
    '''add an entry to the log'''
    joe = JoePosition(latlon, frame_time, r, pos, image_filename)
    self.log.write(cPickle.dumps(joe, protocol=cPickle.HIGHEST_PROTOCOL))
    self.log.flush()
        
  def add_regions(self, frame_time, regions, pos, image_filename):
    '''add a set of regions to the log, applying geo-referencing.
    Return a list of (lat,lon) tuples for the positions of the regions
    '''
    ret = []
    for r in regions:
      latlon = cuav_util.gps_position_from_image_region(r, pos)
      if latlon is not None:
        (lat, lon) = latlon
        self.add((lat,lon), frame_time, r, pos, image_filename)
        ret.append((lat,lon))
      else:
        ret.append((None,None))
    return ret
        
            
class JoeIterator():
  '''an iterator for a joe.log'''
  def __init__(self, filename):
    self.log = open(filename, "r")

  def __iter__(self):
    return self

  def next(self):
    try:
      joe = cPickle.load(self.log)
    except Exception:
      raise StopIteration
    return joe
  

if __name__ == "__main__":
  joelog = JoeIterator(sys.argv[1])
  for joe in joelog:
    print joe
    
