#!/usr/bin/python

from numpy import array,zeros
import chameleon, cv, time

colour = 0
depth = 8
try:
  h = chameleon.open(1, depth)
  colour = 1
except chameleon.error:
  h = chameleon.open(0, depth)
  colour = 0

print("Found camera: colour=%u GUID=%x" % (colour, chameleon.guid(h)))
im = zeros((960,1280),dtype='uint8')

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
  mat = cv.fromarray(im)
  img = cv.GetImage(mat)
  if colour == 1:
    color_img = cv.CreateImage((1280,960), 8, 3)
#    cv.CvtColor(img, color_img, cv.CV_BayerGR2BGR)
#    img_640 = cv.CreateImage((640,480), 8, 3)
#    cv.Resize(color_img, img_640)
#   cv.SaveImage('tmp/i%u_full.jpg' % i, color_img)
  else:
    img_640 = cv.CreateImage((640,480), 8, 1)
#    cv.Resize(img, img_640)
#  cv.ShowImage('Viewer', img_640)
#  cv.SaveImage('tmp/i%u.jpg' % i, img_640)
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

