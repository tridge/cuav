#!/usr/bin/python
"""
benchmark the base cuav operations
"""

import numpy, os, time, cv2, sys, cPickle, pickle
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
        
    #if not hasattr(scanner, 'jpeg_compress'):
    #    return
  
    #for quality in [30, 40, 50, 60, 70, 80, 90, 95]:
    #    t0 = time.time()
    #    for i in range(repeat):
    #        jpeg = cPickle.dumps(ImagePacket(time.time(), scanner.jpeg_compress(colour, quality)),
    #                       protocol=cPickle.HIGHEST_PROTOCOL)
    #    t1 = time.time()
    #    print('jpeg full quality %u: %.1f fps  %u bytes' % (quality, repeat/(t1-t0), len(bytes(jpeg))))

    #for quality in [30, 40, 50, 60, 70, 80, 90, 95]:
    #    t0 = time.time()
    #    for i in range(repeat):
    #        img2 = cv.fromarray(im_full)
    #        jpeg = cPickle.dumps(ImagePacket(time.time(), 
    #                                   cv.EncodeImage('.jpeg', img2, [cv.CV_IMWRITE_JPEG_QUALITY,quality]).tostring()), protocol=cPickle.HIGHEST_PROTOCOL)
    #    t1 = time.time()
    #    print('EncodeImage full quality %u: %.1f fps  %u bytes' % (quality, repeat/(t1-t0), len(bytes(jpeg))))

    #for quality in [30, 40, 50, 60, 70, 80, 90, 95]:
    #    t0 = time.time()
    #    for i in range(repeat):
    #        jpeg = cPickle.dumps(ImagePacket(time.time(), scanner.jpeg_compress(colour_half, quality)),
    #                       protocol=cPickle.HIGHEST_PROTOCOL)
    #    t1 = time.time()
    #    print('jpeg half quality %u: %.1f fps  %u bytes' % (quality, repeat/(t1-t0), len(bytes(jpeg))))

    #for thumb_size in [10, 20, 40, 60, 80, 100]:
    #    thumb = numpy.zeros((thumb_size,thumb_size,3),dtype='uint8')
    #    t0 = time.time()
    #    for i in range(repeat):
    #        scanner.rect_extract(colour, thumb, 0, 0)
    #        jpeg = cPickle.dumps(ImagePacket(time.time(), scanner.jpeg_compress(thumb, 85)),
    #                       protocol=cPickle.HIGHEST_PROTOCOL)
    #    t1 = time.time()
    #    print('thumb %u quality 85: %.1f fps  %u bytes' % (thumb_size, repeat/(t1-t0), len(bytes(jpeg))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cuav benchmarking test")
    parser.add_argument("file", default=None, help="Image file to test with")
    parser.add_argument("--repeat", type=int, default=100, help="repeat count")

    args = parser.parse_args()

    process(args.file, args.repeat)
