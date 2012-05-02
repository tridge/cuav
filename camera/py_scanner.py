#!/usr/bin/python

import chameleon, cv, time, sys, numpy, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
import scanner

colour = 0
depth = 8
try:
  h = chameleon.open(1, depth)
  colour = 1
except chameleon.error:
  h = chameleon.open(0, depth)
  colour = 0

print("Found camera: colour=%u GUID=%x" % (colour, chameleon.guid(h)))
im = numpy.zeros((960,1280),dtype='uint8')
im_640 = numpy.zeros((480,640,3),dtype='uint8')
im_marked = numpy.zeros((480,640,3),dtype='uint8')

cv.NamedWindow('Viewer')

tstart = time.time()

i=0
while True:
  try:
    chameleon.trigger(h)
    (shutter, ftime) = chameleon.capture(h, im)
  except chameleon.error, msg:
    print('failed to capture', msg)
    continue
  scanner.debayer(im, im_640)
  regions = scanner.scan(im_640, im_marked)
  if len(regions) > 0:
    print("Found %u regions" % len(regions))
    for r in regions:
      (minx, miny, maxx, maxy) = r
      print(minx, miny, maxx, maxy)

  mat = cv.fromarray(im_marked)
  img = cv.GetImage(mat)

  # compress using neon-accelerated compressor, and write to a file
  jpeg = scanner.jpeg_compress(im_marked)
  jfile = open('tmp/i%u.jpg' % i, "w")
  jfile.write(jpeg)
  jfile.close()

  cv.ShowImage('Viewer', img)
  i += 1

  if i % 10 == 0:
    tdiff = time.time() - tstart
    print("%.1f fps" % (10/tdiff));
    tstart = time.time()

  key = cv.WaitKey(1)
  if key == -1:
    continue
  if key == ord('q'):
    break

chameleon.close(h)
cv.DestroyWindow('Viewer')

