
#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "chameleon.h"
#include "chameleon_util.h"

static PyObject *ChameleonError;

#define NUM_CAMERA_HANDLES 2

#ifndef Py_RETURN_NONE
#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

static struct chameleon_camera* cameras[NUM_CAMERA_HANDLES] = {
  NULL, NULL
};

float shutters[NUM_CAMERA_HANDLES] = {
  0.0, 0.0
};

static PyObject *
chameleon_open(PyObject *self, PyObject *args)
{
	int colour = 0;
	int depth = 0;
	int sts = -1;

	if (!PyArg_ParseTuple(args, "ii", &colour, &depth))
		return NULL;

	int i = 0;
	for (i = 0; i < NUM_CAMERA_HANDLES; ++i) {
		if (cameras[i] == NULL) {
			struct chameleon_camera *cam = open_camera(colour, depth);
			if (cam != NULL) {
				cameras[i] = cam;
				sts = i;
				break;
			} else {
				break;
			}
		}
	}
	if (i == NUM_CAMERA_HANDLES) {
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
	int status;
	struct chameleon_camera* cam = NULL;
	if (!PyArg_ParseTuple(args, "i", &handle))
		return NULL;

	if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
		cam = cameras[handle];
	} else {
		PyErr_SetString(ChameleonError, "Invalid handle");
		return NULL;
	}

	Py_BEGIN_ALLOW_THREADS;
	status = trigger_capture(cam, shutters[handle]);
	Py_END_ALLOW_THREADS;

	if (status < 0) {
		PyErr_SetString(ChameleonError, "Failed to capture");
		return NULL;
	}

	Py_RETURN_NONE;
}

static PyObject *
chameleon_capture(PyObject *self, PyObject *args)
{
	int handle = -1;
	struct chameleon_camera* cam = NULL;
	PyArrayObject* array = NULL;
	if (!PyArg_ParseTuple(args, "iO", &handle, &array))
		return NULL;

	if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
		cam = cameras[handle];
	} else {
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
	int status;

	Py_BEGIN_ALLOW_THREADS;
	status = capture_wait(cam, &shutters[handle], buf, stride, stride*h);
	Py_END_ALLOW_THREADS;
	
	if (status < 0) {
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
		close_camera(cameras[handle]);
		cameras[handle] = NULL;
		sts = 0;
	} else {
		PyErr_SetString(ChameleonError, "Invalid handle");
		return NULL;
	}

	if (sts < 0) {
		PyErr_SetString(ChameleonError, "Failed to close device");
		return NULL;
	}
	
	Py_RETURN_NONE;
}


static PyObject *
chameleon_guid(PyObject *self, PyObject *args)
{
	int handle;
	if (!PyArg_ParseTuple(args, "i", &handle))
		return NULL;

	if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
		return PyLong_FromLong(cameras[handle]->guid);
	}
	PyErr_SetString(ChameleonError, "invalid handle");
	return NULL;
}

/* low level save routine */
static int _save_pgm(const char *filename, unsigned w, unsigned h, unsigned stride,
		     const char *data)
{
	int fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC, 0644);
	if (fd == -1) {
		return -1;
	}
	dprintf(fd,"P5\n%u %u\n%u\n", 
		w, h, stride==1?255:65535);
	if (write(fd, data, h*stride) != h*stride) {
		close(fd);
		return -1;
	}
	close(fd);
	return 0;
}

/*
  save a pgm image 
 */
static PyObject *
save_pgm(PyObject *self, PyObject *args)
{
	int handle, status;
	const char *filename;
	unsigned w, h, stride;
	PyArrayObject* array = NULL;

	if (!PyArg_ParseTuple(args, "isO", &handle, &filename, &array))
		return NULL;

	if (!(handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle])) {
		PyErr_SetString(ChameleonError, "invalid handle");
		return NULL;
	}

	w = PyArray_DIM(array, 1);
	h = PyArray_DIM(array, 0);
	stride = PyArray_STRIDE(array, 0);

	Py_BEGIN_ALLOW_THREADS;
	status = _save_pgm(filename, w, h, stride, PyArray_DATA(array));
	Py_END_ALLOW_THREADS;
	if (status != 0) {
		PyErr_SetString(ChameleonError, "pgm save failed");
		return NULL;
	}
	Py_RETURN_NONE;
}


static PyMethodDef ChameleonMethods[] = {
  {"open", chameleon_open, METH_VARARGS, "Open a lizard like device. Returns handle"},
  {"close", chameleon_close, METH_VARARGS, "Close device."},
  {"trigger", chameleon_trigger, METH_VARARGS, "Trigger capture of an image"},
  {"capture", chameleon_capture, METH_VARARGS, "Capture an image"},
  {"guid", chameleon_guid, METH_VARARGS, "camera GUID"},
  {"save_pgm", save_pgm, METH_VARARGS, "save to a PGM"},
  {NULL, NULL, 0, NULL}        /* Terminus */
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

