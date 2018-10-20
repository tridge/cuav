#!/usr/bin/python
"""
benchmark the base cuav operations
"""

import numpy, os, time, cv2, sys
import argparse

from cuav.image import scanner
from cuav.lib import cuav_util


class ImagePacket:
    '''a jpeg image sent to the ground station'''
    def __init__(self, frame_time, jpeg):
        self.frame_time = frame_time
        self.jpeg = jpeg

def process(filename, repeat):
    '''process one file'''
    colour = cv2.imread(filename)
    colour_half = cv2.resize(colour, (0,0), fx=0.5, fy=0.5) 

    t0 = time.time()
    for i in range(repeat):
        cv2.cvtColor(colour, cv2.COLOR_RGB2HSV)
    t1 = time.time()
    if t1 > t0:
        print('RGB2HSV_full: %.1f fps' % (repeat/(t1-t0)))
    else:
        print('RGB2HSV_full: (inf) fps')

    t0 = time.time()
    for i in range(repeat):
        cv2.cvtColor(colour_half, cv2.COLOR_RGB2HSV)
    t1 = time.time()
    if t1 > t0:
        print('RGB2HSV_half: %.1f fps' % (repeat/(t1-t0)))
    else:
        print('RGB2HSV_half: (inf) fps')
        
    t0 = time.time()
    for i in range(repeat):
        thumb = numpy.empty((100,100,3),dtype='uint8')
        scanner.rect_extract(colour, thumb, 120, 125)
    t1 = time.time()
    if t1 > t0:
        print('rect_extract: %.1f fps' % (repeat/(t1-t0)))
    else:
        print('rect_extract: (inf) fps')
        
    t0 = time.time()
    for i in range(repeat):
        thumb = cuav_util.SubImage(colour, (120,125,100,100))
    t1 = time.time()
    if t1 > t0:
        print('SubImage: %.1f fps' % (repeat/(t1-t0)))
    else:
        print('SubImage: (inf) fps')

    #t0 = time.time()
    #for i in range(repeat):
    #    scanner.downsample(im_full, im_640)
    #t1 = time.time()
    #print('downsample: %.1f fps' % (repeat/(t1-t0)))

    t0 = time.time()
    for i in range(repeat):
        scanner.scan(colour_half)
    t1 = time.time()
    if t1 > t0:
        print('scan: %.1f fps' % (repeat/(t1-t0)))
    else:
        print('scan: (inf) fps')
        
    t0 = time.time()
    for i in range(repeat):
        scanner.scan(colour)
    t1 = time.time()
    if t1 > t0:
        print('scan_full: %.1f fps' % (repeat/(t1-t0)))
    else:
        print('scan_full: (inf) fps')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cuav benchmarking test")
    parser.add_argument("file", default=None, help="Image file to test with")
    parser.add_argument("--repeat", type=int, default=100, help="repeat count")

    args = parser.parse_args()

    process(args.file, args.repeat)
