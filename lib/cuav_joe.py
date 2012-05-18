#!/usr/bin/python
'''
object to hold Joe positions
We store a pickled list of these objects in joe.log
Andrew Tridgell
May 2012
'''

import os, sys, cuav_util, mav_position, cPickle


class JoePosition():
  '''a Joe position'''
  def __init__(self, latlon, frame_time,
               pos, image_filename):
      self.latlon = latlon
      self.frame_time = frame_time
      self.pos = pos
      self.image_filename = image_filename

      
class JoeLog():
    '''a Joe position logger'''
    def __init__(self, filename, append=True):
        self.filename = filename
        self.log = open(filename, "w+" if append else "w")
        
    def add(self, latlon, frame_time, pos, image_filename):
        '''add an entry to the log'''
        joe = JoePosition(latlon, frame_time, pos, image_filename)
        self.log.write(cPickle.dumps(joe, protocol=cPickle.HIGHEST_PROTOCOL))
        self.log.flush()
        
    def add_regions(self, frame_time, regions, pos, image_filename):
        '''add a set of regions to the log, applying geo-referencing'''
        for r in regions:
            (lat, lon) = cuav_util.gps_position_from_image_region(r, pos)
            self.add((lat,lon), frame_time, pos, image_filename)
            
