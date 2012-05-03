#!/usr/bin/python

import chameleon, cv, time, sys, os, numpy
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
  if colour == 1:
    img_colour = numpy.zeros((480,640,3),dtype='uint8')
    scanner.debayer(im, img_colour)
    img_640 = cv.GetImage(cv.fromarray(img_colour))
  else:
    img_640 = cv.CreateImage((640,480), 8, 1)
    mat = cv.fromarray(im)
    img = cv.GetImage(mat)
    cv.Resize(img, img_640)
  cv.ShowImage('Viewer', img_640)
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

