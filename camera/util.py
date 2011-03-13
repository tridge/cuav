'''common utility functions'''

import numpy, cv

class PGMError(Exception):
	'''PGMLink error class'''
	def __init__(self, msg):
            Exception.__init__(self, msg)


class PGM(object):
    '''16 bit 1280x960 PGM image handler'''
    def __init__(self, filename):
        self.filename = filename
        
        f = open(filename, mode='r')
        fmt = f.readline()
        if fmt.strip() != 'P5':
            raise PGMError('Expected P5 image in %s' % filename)
        dims = f.readline()
        if dims.strip() != '1280 960':
            raise PGMError('Expected 1280x960 image in %s' % filename)
        line = f.readline()
        if line[0] == '#':
            self.comment = line
            line = f.readline()
        if line.strip() != '65535':
            raise PGMError('Expected 16 bit image image in %s - got %s' % (filename, line.strip()))
        ofs = f.tell()
        f.close()
        a = numpy.memmap(filename, dtype='uint16', mode='c', order='C', shape=(960,1280), offset=ofs)
        self.array = a.byteswap(True)
        self.img = cv.CreateImageHeader((1280, 960), 16, 1)
        cv.SetData(self.img, self.array.tostring(), self.array.dtype.itemsize*1*1280)

