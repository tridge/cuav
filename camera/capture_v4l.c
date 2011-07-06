/*
 *  V4L2 video capture example
 *
 *  This program can be used and distributed without restrictions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <asm/types.h>          /* for videodev2.h */

#include <linux/videodev2.h>

#define CLEAR(x) memset (&(x), 0, sizeof (x))

typedef enum {
  IO_METHOD_READ,
  IO_METHOD_MMAP,
} io_method;

struct buffer {
  void *         start;
  size_t         length;
  struct timeval timestamp;
};

struct device {
  char*              dev_name;
  char*              base_name;
  size_t             frame_cnt;
  io_method          io;
  int                fd;
  struct buffer*     buffers;
  unsigned int       n_buffers;
  struct v4l2_format fmt;
};

#define MAX_DEVICES 4

static struct device devices[MAX_DEVICES] =
{
  {NULL, NULL, 0, IO_METHOD_MMAP, -1, NULL, 0},
  {NULL, NULL, 0, IO_METHOD_MMAP, -1, NULL, 0},
  {NULL, NULL, 0, IO_METHOD_MMAP, -1, NULL, 0},
  {NULL, NULL, 0, IO_METHOD_MMAP, -1, NULL, 0}
};

static unsigned int n_devices = 1;

static void
errno_exit(const char *           s)
{
  fprintf (stderr, "%s error %d, %s\n",
           s, errno, strerror (errno));

  exit (EXIT_FAILURE);
}

static int
xioctl(int                    fd,
       int                    request,
       void *                 arg)
{
  int r;

  do r = ioctl(fd, request, arg);
  while (-1 == r && EINTR == errno);

  return r;
}

static void
process_image(struct device* dev, size_t idx)
{
  fputc ('.', stdout);
  void* p = dev->buffers[idx].start;
  size_t len = dev->buffers[idx].length;

  char fname[256];
  snprintf(fname, 255, dev->base_name, dev->frame_cnt);
  // display -size 640x480 -depth 8 -colorspace RGB -sampling-factor 4:2:0 -interlace plane foo00000.yuv
  int f = open(fname, O_CREAT|O_TRUNC|O_RDWR, S_IRUSR|S_IWUSR);
  switch(dev->fmt.fmt.pix.pixelformat)
  {
  case V4L2_PIX_FMT_YUV420:
    assert(len <= dev->buffers[idx].length);
    size_t len = 3*dev->fmt.fmt.pix.width*dev->fmt.fmt.pix.height/2;
    write(f, p, len);
    break;
  default:
    break;
  }
  dev->frame_cnt++;
  fsync(f);
  close(f);

  fflush (stdout);
}

static int
read_frame(struct device* dev)
{
  struct v4l2_buffer buf;

  switch (dev->io) {
  case IO_METHOD_READ:
    if (-1 == read(dev->fd, dev->buffers[0].start, dev->buffers[0].length)) {
      switch (errno) {
      case EAGAIN:
        return 0;

      case EIO:
        /* Could ignore EIO, see spec. */

        /* fall through */

      default:
        errno_exit("read");
      }
    }

    process_image(dev, 0);

    break;

  case IO_METHOD_MMAP:
    CLEAR(buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(dev->fd, VIDIOC_DQBUF, &buf)) {
      switch (errno) {
      case EAGAIN:
        return 0;

      case EIO:
        /* Could ignore EIO, see spec. */

        /* fall through */

      default:
        errno_exit("VIDIOC_DQBUF");
      }
    }

    printf("%ld.%03ld\n", buf.timestamp.tv_sec, buf.timestamp.tv_usec/1000);

    assert(buf.index < dev->n_buffers);

    process_image(dev, buf.index);

    struct timeval* tv = &(dev->buffers[buf.index].timestamp);
    gettimeofday(tv, NULL);
    printf("%ld.%03ld\n", tv->tv_sec, tv->tv_usec/1000);
    if (-1 == xioctl(dev->fd, VIDIOC_QBUF, &buf))
      errno_exit("VIDIOC_QBUF");

    break;
  }

  return 1;
}

static void
mainloop(void)
{
  unsigned int count;

  count = 100;

  while (count-- > 0) {
    for (;;) {
      fd_set fds;
      struct timeval tv;
      int r;

      int i;
      for (i = 0; i < n_devices; ++i)
      {
        FD_ZERO(&fds);
        FD_SET(devices[i].fd, &fds);

        /* Timeout. */
        tv.tv_sec = 2;
        tv.tv_usec = 0;

        r = select(devices[i].fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r) {
          if (EINTR == errno)
            continue;

          errno_exit("select");
        }

        if (0 == r) {
          fprintf(stderr, "select timeout\n");
          exit (EXIT_FAILURE);
        }
      }

      for (i = 0; i < n_devices; ++i)
      {
        read_frame(&devices[i]);
      }

      /* EAGAIN - continue select loop. */
    }
  }
}

static void
stop_capturing(struct device* dev)
{
  enum v4l2_buf_type type;

  switch (dev->io) {
  case IO_METHOD_READ:
    /* Nothing to do. */
    break;

  case IO_METHOD_MMAP:
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (-1 == xioctl (dev->fd, VIDIOC_STREAMOFF, &type))
      errno_exit ("VIDIOC_STREAMOFF");

    break;
  }
}

static void
start_capturing(struct device* dev)
{
  unsigned int i;
  enum v4l2_buf_type type;

  switch (dev->io) {
  case IO_METHOD_READ:
    /* Nothing to do. */
    break;

  case IO_METHOD_MMAP:
    for (i = 0; i < dev->n_buffers; ++i) {
      struct v4l2_buffer buf;

      CLEAR (buf);

      buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory      = V4L2_MEMORY_MMAP;
      buf.index       = i;

      if (-1 == xioctl (dev->fd, VIDIOC_QBUF, &buf))
        errno_exit ("VIDIOC_QBUF");
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (-1 == xioctl (dev->fd, VIDIOC_STREAMON, &type))
      errno_exit ("VIDIOC_STREAMON");

    break;
  }
}

static void
uninit_device(struct device* dev)
{
  unsigned int i;

  switch (dev->io) {
  case IO_METHOD_READ:
    free (dev->buffers[0].start);
    break;

  case IO_METHOD_MMAP:
    for (i = 0; i < dev->n_buffers; ++i)
      if (-1 == munmap (dev->buffers[i].start, dev->buffers[i].length))
        errno_exit ("munmap");
    break;
  }

  free (dev->buffers);
}

static void
init_read(struct device* dev, unsigned int buffer_size)
{
  dev->buffers = calloc (1, sizeof (struct buffer));

  if (!dev->buffers) {
    fprintf (stderr, "Out of memory\n");
    exit (EXIT_FAILURE);
  }

  dev->buffers[0].length = buffer_size;
  dev->buffers[0].start = malloc (buffer_size);

  if (!dev->buffers[0].start) {
    fprintf (stderr, "Out of memory\n");
    exit (EXIT_FAILURE);
  }
}

static void
init_mmap(struct device* dev)
{
  struct v4l2_requestbuffers req;

  CLEAR (req);

  req.count               = 4;
  req.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory              = V4L2_MEMORY_MMAP;

  if (-1 == xioctl (dev->fd, VIDIOC_REQBUFS, &req)) {
    if (EINVAL == errno) {
      fprintf (stderr, "%s does not support "
               "memory mapping\n", dev->dev_name);
      exit (EXIT_FAILURE);
    } else {
      errno_exit ("VIDIOC_REQBUFS");
    }
  }

  if (req.count < 2) {
    fprintf (stderr, "Insufficient buffer memory on %s\n",
             dev->dev_name);
    exit (EXIT_FAILURE);
  }

  dev->buffers = calloc (req.count, sizeof(struct buffer));

  if (!dev->buffers) {
    fprintf (stderr, "Out of memory\n");
    exit (EXIT_FAILURE);
  }

  for (dev->n_buffers = 0; dev->n_buffers < req.count; ++dev->n_buffers) {
    struct v4l2_buffer buf;

    CLEAR (buf);

    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = dev->n_buffers;

    if (-1 == xioctl (dev->fd, VIDIOC_QUERYBUF, &buf))
      errno_exit ("VIDIOC_QUERYBUF");

    dev->buffers[dev->n_buffers].length = buf.length;
    dev->buffers[dev->n_buffers].start =
      mmap (NULL /* start anywhere */,
            buf.length,
            PROT_READ | PROT_WRITE /* required */,
            MAP_SHARED /* recommended */,
            dev->fd, buf.m.offset);

    if (MAP_FAILED == dev->buffers[dev->n_buffers].start)
      errno_exit ("mmap");
  }
}

static void
init_device(struct device* dev)
{
  struct v4l2_capability cap;
  struct v4l2_cropcap cropcap;
  struct v4l2_crop crop;
  unsigned int min;

  if (-1 == xioctl (dev->fd, VIDIOC_QUERYCAP, &cap)) {
    if (EINVAL == errno) {
      fprintf (stderr, "%s is no V4L2 device\n",
               dev->dev_name);
      exit (EXIT_FAILURE);
    } else {
      errno_exit ("VIDIOC_QUERYCAP");
    }
  }

  if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    fprintf (stderr, "%s is no video capture device\n",
             dev->dev_name);
    exit (EXIT_FAILURE);
  }

  switch (dev->io) {
  case IO_METHOD_READ:
    if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
      fprintf (stderr, "%s does not support read i/o\n",
               dev->dev_name);
      exit (EXIT_FAILURE);
    }

    break;

  case IO_METHOD_MMAP:
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
      fprintf (stderr, "%s does not support streaming i/o\n",
               dev->dev_name);
      exit (EXIT_FAILURE);
    }

    break;
  }


  /* Select video input, video standard and tune here. */


  CLEAR (cropcap);

  cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if (0 == xioctl(dev->fd, VIDIOC_CROPCAP, &cropcap)) {
    crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    crop.c = cropcap.defrect; /* reset to default */

    if (-1 == xioctl(dev->fd, VIDIOC_S_CROP, &crop)) {
      switch (errno) {
      case EINVAL:
        /* Cropping not supported. */
        break;
      default:
        /* Errors ignored. */
        break;
      }
    }
  } else {
    /* Errors ignored. */
  }


  CLEAR(dev->fmt);

  dev->fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  dev->fmt.fmt.pix.width       = 640;
  dev->fmt.fmt.pix.height      = 480;
  dev->fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
  dev->fmt.fmt.pix.field       = V4L2_FIELD_NONE;

  if (-1 == xioctl (dev->fd, VIDIOC_S_FMT, &dev->fmt))
    errno_exit("VIDIOC_S_FMT");

  /* Note VIDIOC_S_FMT may change width and height. */

  /* Buggy driver paranoia. */
  min = dev->fmt.fmt.pix.width * 2;
  if (dev->fmt.fmt.pix.bytesperline < min)
    dev->fmt.fmt.pix.bytesperline = min;
  min = dev->fmt.fmt.pix.bytesperline * dev->fmt.fmt.pix.height;
  if (dev->fmt.fmt.pix.sizeimage < min)
    dev->fmt.fmt.pix.sizeimage = min;

  switch (dev->io) {
  case IO_METHOD_READ:
    init_read(dev, dev->fmt.fmt.pix.sizeimage);
    break;

  case IO_METHOD_MMAP:
    init_mmap(dev);
    break;
  }
}

static void
close_device(struct device* dev)
{
  if (-1 == close (dev->fd))
    errno_exit("close");

  dev->fd = -1;
}

static void
open_device(struct device* dev)
{
  struct stat st;

  if (-1 == stat(dev->dev_name, &st)) {
    fprintf(stderr, "Cannot identify '%s': %d, %s\n",
            dev->dev_name, errno, strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (!S_ISCHR(st.st_mode)) {
    fprintf(stderr, "%s is no device\n", dev->dev_name);
    exit(EXIT_FAILURE);
  }

  dev->fd = open(dev->dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

  if (-1 == dev->fd) {
    fprintf(stderr, "Cannot open '%s': %d, %s\n",
            dev->dev_name, errno, strerror(errno));
    exit(EXIT_FAILURE);
  }
}

static void
usage(FILE*  fp,
      int    argc,
      char** argv)
{
  fprintf(fp,
          "Usage: %s [options]\n\n"
          "Options:\n"
          "-d | --device name   Video device name [/dev/video]\n"
          "-h | --help          Print this message\n"
          "-m | --mmap          Use memory mapped buffers\n"
          "-r | --read          Use read() calls\n"
          "-b | --base          Base file name\n"
          "",
          argv[0]);
}

static const char short_options [] = "d:b:hmru";

static const struct option
long_options [] = {
  { "device",     required_argument,      NULL,           'd' },
  { "base",       required_argument,      NULL,           'b' },
  { "help",       no_argument,            NULL,           'h' },
  { "mmap",       no_argument,            NULL,           'm' },
  { "read",       no_argument,            NULL,           'r' },
  { 0, 0, 0, 0 }
};

int
main(int argc,
     char** argv)
{
  devices[n_devices-1].dev_name = "/dev/video0";

  int dev_cnt = 0;
  for (;;) {
    int index;
    int c;

    c = getopt_long(argc, argv,
                    short_options, long_options,
                    &index);

    if (-1 == c)
      break;

    switch (c) {
    case 0: /* getopt_long() flag */
      break;

    case 'd':
      if (dev_cnt == MAX_DEVICES)
      {
        fprintf(stderr, "too many devices\n");
        exit(EXIT_FAILURE);
      }
      if (dev_cnt)
      {
        n_devices++;
      }
      dev_cnt++;
      devices[n_devices-1].dev_name = optarg;
      break;

    case 'h':
      usage(stdout, argc, argv);
      exit(EXIT_SUCCESS);

    case 'm':
      devices[n_devices-1].io = IO_METHOD_MMAP;
      break;

    case 'r':
      devices[n_devices-1].io = IO_METHOD_READ;
      break;

    case 'b':
      devices[n_devices-1].base_name = optarg;
      break;

    default:
      usage(stderr, argc, argv);
      exit(EXIT_FAILURE);
    }
  }

  int i;
  for (i = 0; i< n_devices; ++i)
  {
    open_device(&devices[i]);
    init_device(&devices[i]);
  }

  for (i = 0; i< n_devices; ++i)
  {
    start_capturing(&devices[i]);
  }

  mainloop();

  for (i = 0; i< n_devices; ++i)
  {
    stop_capturing(&devices[i]);
    uninit_device(&devices[i]);
    close_device(&devices[i]);
  }

  exit(EXIT_SUCCESS);

  return 0;
}
