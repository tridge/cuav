#!/usr/bin/python
'''
object to hold Joe positions
We store a pickled list of these objects in joe.log
Andrew Tridgell
May 2012
'''

import os, sys, pickle, time
from cuav.lib import cuav_util, mav_position


class JoePosition():
  '''a Joe position'''
  def __init__(self, latlon, frame_time, r,
               pos, image_filename, thumb_filename):
      self.latlon = latlon
      self.frame_time = frame_time
      self.pos = pos
      self.r = r
      self.image_filename = image_filename
      self.thumb_filename = thumb_filename

  def rawname(self):
    '''return raw filename'''
    return '%s' % cuav_util.frame_time(self.frame_time)
  
  def __str__(self):
    return 'JoePosition(lat=%f lon=%f %s %s %s %s %s %s)' % (self.latlon[0], self.latlon[1],
                                                             self.pos, self.image_filename,
                                                             self.thumb_filename,
                                                             str(getattr(self,'r','')),
                                                             time.asctime(time.localtime(self.frame_time)),
                                                             self.rawname())
      
class JoeLog():
  '''a Joe position logger'''
  def __init__(self, filename, append=True):
    self.filename = filename
    if filename is not None:
      self.log = filename #open(filename, "ab" if append else "wb")
      if not append:
        try:
          os.remove(self.filename)
        except:
          pass
    else:
      self.log = None
        
  def add(self, latlon, frame_time, r, pos, image_filename, thumb_filename):
    '''add an entry to the log'''
    joe = JoePosition(latlon, frame_time, r, pos, image_filename, thumb_filename)
    if self.log is not None:
      with open(self.filename, "ab") as f:
        f.write(pickle.dumps(joe, protocol=pickle.HIGHEST_PROTOCOL))
        f.flush()
        
  def add_regions(self, frame_time, regions, pos, image_filename, thumb_filename=None):
    '''add a set of regions to the log, applying geo-referencing.
    Add latlon attribute to regions
    '''
    ret = []
    for r in regions:
        self.add(r.latlon, frame_time, r, pos, image_filename, thumb_filename)
    return ret
        
            
class JoeIterator():
  '''an iterator for a joe.log'''
  def __init__(self, filename):
    self.joes = []
    
    in_s = open(filename, 'rb')
    try:
      # Read the data
      while True:
        try:
          o = pickle.load(in_s)
          self.joes.append(o)
        except EOFError:
          break
    finally:
      in_s.close()

  def getjoes(self):
    return self.joes
  

