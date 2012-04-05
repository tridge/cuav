
#include <python2.6/Python.h>
#include <python2.6/numpy/arrayobject.h>
#include "chameleon.h"
#include "chameleon_util.h"

static PyObject *
chameleon_open(PyObject *self, PyObject *args);
static PyObject *
chameleon_close(PyObject *self, PyObject *args);
static PyObject *
chameleon_trigger(PyObject *self, PyObject *args);
static PyObject *
chameleon_capture(PyObject *self, PyObject *args);

static PyObject *ChameleonError;

static PyMethodDef ChameleonMethods[] = {
  {"open", chameleon_open, METH_VARARGS, "Open a lizard like device. Returns handle"},
  {"close", chameleon_close, METH_VARARGS, "Close device."},
  {"trigger", chameleon_trigger, METH_VARARGS, "Trigger capture of an image"},
  {"capture", chameleon_capture, METH_VARARGS, "Capture an image"},
  {NULL, NULL, 0, NULL}        /* Terminus */
};

#define NUM_CAMERA_HANDLES 2

static struct chameleon_camera* cameras[NUM_CAMERA_HANDLES] = {
  NULL, NULL
};

float shutters[NUM_CAMERA_HANDLES] = {
  0.0, 0.0
};

PyMODINIT_FUNC
initchameleon(void)
{
  PyObject *m;

  m = Py_InitModule("chameleon", ChameleonMethods);
  if (m == NULL)
    return;

  ChameleonError = PyErr_NewException("chameleon.error", NULL, NULL);
  Py_INCREF(ChameleonError);
  PyModule_AddObject(m, "error", ChameleonError);
}

static PyObject *
chameleon_open(PyObject *self, PyObject *args)
{
  int colour = 0;
  int depth = 0;
  int sts = -1;

  if (!PyArg_ParseTuple(args, "ii", &colour, &depth))
    return NULL;

  int i = 0;
  for (i = 0; i < NUM_CAMERA_HANDLES; ++i)
  {
    if (cameras[i] == NULL)
    {
      struct chameleon_camera* cam = open_camera(colour, depth);
      if (cam != NULL)
      {
        cameras[i] = cam;
        sts = i;
        break;
      }
      else
      {
        break;
      }
    }
  }
  if (i == NUM_CAMERA_HANDLES)
  {
    PyErr_SetString(ChameleonError, "No camera handles available");
    return NULL;
  }
  if (sts < 0) {
    PyErr_SetString(ChameleonError, "Failed to open device");
    return NULL;
  }
  return PyLong_FromLong(sts);
}

static PyObject *
chameleon_trigger(PyObject *self, PyObject *args)
{
  int handle = -1;
  struct chameleon_camera* cam = NULL;
  PyArrayObject* array;
  if (!PyArg_ParseTuple(args, "i", &handle, &array))
    return NULL;

  if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
    cam = cameras[handle];
  }
  else {
    PyErr_SetString(ChameleonError, "Invalid handle");
    return NULL;
  }

  int sts = trigger_capture(cam, shutters[handle]);

  if (sts < 0) {
    PyErr_SetString(ChameleonError, "Failed to capture");
    return NULL;
  }

  return PyLong_FromLong(sts);
}

static PyObject *
chameleon_capture(PyObject *self, PyObject *args)
{
  int handle = -1;
  struct chameleon_camera* cam = NULL;
  PyArrayObject* array;
  if (!PyArg_ParseTuple(args, "iO", &handle, &array))
    return NULL;

  if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
    cam = cameras[handle];
  }
  else {
    PyErr_SetString(ChameleonError, "Invalid handle");
    return NULL;
  }

  int ndim = PyArray_NDIM(array);
  //printf("ndim=%d\n", ndim);
  if (ndim != 2){
    PyErr_SetString(ChameleonError, "Array has invalid number of dimensions");
    return NULL;
  }

  int w = PyArray_DIM(array, 1);
  int h = PyArray_DIM(array, 0);
  int stride = PyArray_STRIDE(array, 0);
  //printf("w=%d, h=%d, stride=%d\n", w,h,stride);
  if (w != 1280 || h != 960){
    PyErr_SetString(ChameleonError, "Invalid array dimensions should be 960x1280");
    return NULL;
  }

  void* buf = PyArray_DATA(array);
  struct timeval tv;
  int sts = capture_wait(cam, &shutters[handle], buf, stride, stride*h);

  if (sts < 0) {
    PyErr_SetString(ChameleonError, "Failed to capture");
    return NULL;
  }
  int64_t time = (int64_t)tv.tv_sec*1000000LL + (int64_t)tv.tv_usec;
  return Py_BuildValue("fL", shutters[handle], time);
}

static PyObject *
chameleon_close(PyObject *self, PyObject *args)
{
  int sts = -1;
  int handle = -1;
  if (!PyArg_ParseTuple(args, "i", &handle))
    return NULL;

  if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
    chameleon_camera_free(cameras[handle]);
    cameras[handle] = NULL;
    sts = 0;
  }
  else {
    PyErr_SetString(ChameleonError, "Invalid handle");
    return NULL;
  }

  if (sts < 0) {
    PyErr_SetString(ChameleonError, "Failed to close device");
    return NULL;
  }

  return PyLong_FromLong(sts);
}


