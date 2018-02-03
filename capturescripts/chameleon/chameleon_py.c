
#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "include/chameleon.h"
#include "include/chameleon_util.h"
#include <sched.h>

static PyObject *ChameleonError;

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

#define NUM_CAMERA_HANDLES 2

#ifndef Py_RETURN_NONE
#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

#define CHECK_CONTIGUOUS(a) do { if (!PyArray_ISCONTIGUOUS(a)) { \
	PyErr_SetString(ChameleonError, "array must be contiguous"); \
	return NULL; \
	}} while (0)

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
	unsigned short depth = 0;
	unsigned short brightness;
	int sts = -1;
	PyObject *colour_obj;
	
	if (!PyArg_ParseTuple(args, "OHH", &colour_obj, &depth, &brightness))
		return NULL;

	colour = PyObject_IsTrue(colour_obj);

         // try to make the capture thread realtime
        struct sched_param p;
        memset(&p, 0, sizeof(p));
        p.sched_priority = SCHED_FIFO;
        sched_setscheduler(0, SCHED_FIFO, &p);
        
	int i = 0;
	for (i = 0; i < NUM_CAMERA_HANDLES; ++i) {
		if (cameras[i] == NULL) {
			struct chameleon_camera *cam = open_camera(colour, depth, brightness);
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
	bool continuous;
	PyObject *continuous_obj;

	if (!PyArg_ParseTuple(args, "iO", &handle, &continuous_obj))
		return NULL;

	continuous = PyObject_IsTrue(continuous_obj);

	if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
		cam = cameras[handle];
	} else {
		PyErr_SetString(ChameleonError, "Invalid handle");
		return NULL;
	}

	Py_BEGIN_ALLOW_THREADS;
	status = trigger_capture(cam, shutters[handle], continuous);
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
	int timeout_ms = 0;
	struct chameleon_camera* cam = NULL;
	PyArrayObject* array = NULL;
	if (!PyArg_ParseTuple(args, "iiO", &handle, &timeout_ms, &array))
		return NULL;

	CHECK_CONTIGUOUS(array);

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
	int status;
	float frame_time=0;
	uint32_t frame_counter=0;

	Py_BEGIN_ALLOW_THREADS;
	status = capture_wait(cam, &shutters[handle], buf, stride, stride*h, 
			      timeout_ms, &frame_time, &frame_counter);
	Py_END_ALLOW_THREADS;
	
	if (status < 0) {
		PyErr_SetString(ChameleonError, "Failed to capture");
		return NULL;
	}
	return Py_BuildValue("flf", 
			     frame_time, 
			     (long)frame_counter,
			     shutters[handle]);
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


static PyObject *
chameleon_set_brightness(PyObject *self, PyObject *args)
{
	int handle = -1;
	int brightness=0;
	struct chameleon_camera* cam = NULL;

	if (!PyArg_ParseTuple(args, "ii", &handle, &brightness))
		return NULL;

	if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
		cam = cameras[handle];
	} else {
		PyErr_SetString(ChameleonError, "Invalid handle");
		return NULL;
	}

	Py_BEGIN_ALLOW_THREADS;
	camera_set_brightness(cam, brightness);
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}

static PyObject *
chameleon_set_gamma(PyObject *self, PyObject *args)
{
	int handle = -1;
	int gamma=0;
	struct chameleon_camera* cam = NULL;

	if (!PyArg_ParseTuple(args, "ii", &handle, &gamma))
		return NULL;

	if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
		cam = cameras[handle];
	} else {
		PyErr_SetString(ChameleonError, "Invalid handle");
		return NULL;
	}

	Py_BEGIN_ALLOW_THREADS;
	camera_set_gamma(cam, gamma);
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}

static PyObject *
chameleon_set_framerate(PyObject *self, PyObject *args)
{
	int handle = -1;
	int framerate=0;
	struct chameleon_camera* cam = NULL;

	if (!PyArg_ParseTuple(args, "ii", &handle, &framerate))
		return NULL;

	if (handle >= 0 && handle < NUM_CAMERA_HANDLES && cameras[handle]) {
		cam = cameras[handle];
	} else {
		PyErr_SetString(ChameleonError, "Invalid handle");
		return NULL;
	}

	Py_BEGIN_ALLOW_THREADS;
	camera_set_framerate(cam, framerate);
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}

static PyMethodDef ChameleonMethods[] = {
  {"open", (PyCFunction) chameleon_open, METH_VARARGS, "Open a lizard like device. Returns handle"},
  {"close", (PyCFunction) chameleon_close, METH_VARARGS, "Close device."},
  {"trigger", (PyCFunction) chameleon_trigger, METH_VARARGS, "Trigger capture of an image"},
  {"capture", (PyCFunction) chameleon_capture, METH_VARARGS, "Capture an image"},
  {"guid", (PyCFunction) chameleon_guid, METH_VARARGS, "camera GUID"},
  {"set_gamma", (PyCFunction) chameleon_set_gamma, METH_VARARGS, "set gamma"},
  {"set_framerate", (PyCFunction) chameleon_set_framerate, METH_VARARGS, "set framerate in Hz"},
  {"set_brightness", (PyCFunction) chameleon_set_brightness, METH_VARARGS, "set brightness"},
  {NULL}        /* Terminus */
};

#ifdef PY3K
static struct PyModuleDef ChameleonFunc =
{
    PyModuleDef_HEAD_INIT,
    "chameleon", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    ChameleonMethods
};

PyMODINIT_FUNC PyInit_chameleon(void)
{
    PyObject *m;
    m = PyModule_Create(&ChameleonFunc);
    if (m == NULL)
    return;
    
    import_array();

    ChameleonError = PyErr_NewException("chameleon.error", NULL, NULL);
    Py_INCREF(ChameleonError);
    PyModule_AddObject(m, "error", ChameleonError);
    
    return m;
}
#else
PyMODINIT_FUNC
initchameleon(void)
{
  PyObject *m;

  m = Py_InitModule("chameleon", ChameleonMethods);
  if (m == NULL)
    return;

  import_array();

  ChameleonError = PyErr_NewException("chameleon.error", NULL, NULL);
  Py_INCREF(ChameleonError);
  PyModule_AddObject(m, "error", ChameleonError);
}
#endif

