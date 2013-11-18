#!/usr/bin/env python

"""
generic image viewer widget
"""

class CUAVImageViewer():
    '''
    a generic image viewer widget for use in CUAV tools
    '''
    def __init__(self,
                 title='ImageViewer'):
        import multiprocessing
        self.title = title
        self.parent_pipe,self.child_pipe = multiprocessing.Pipe()
        self.close_window = multiprocessing.Event()
        self.close_window.clear()
        self.child = multiprocessing.Process(target=self.child_task)
        self.child.start()

    def child_task(self):
        '''child process - this holds all the GUI elements'''
        import wx, matplotlib
        matplotlib.use('WXAgg')
        app = wx.PySimpleApp()
        app.frame = CUAVImageViewerFrame(state=self)
        app.frame.Show()
        app.MainLoop()

    def close(self):
        '''close the window'''
        self.close_window.set()
        if self.is_alive():
            self.child.join(2)

    def is_alive(self):
        '''check if graph is still going'''
        return self.child.is_alive()

    def ShowImage(self, img):
        '''show an image'''
        imgstring = img.convert("RGB").tostring()
        self.parent_pipe.send((img.size[0], img.size[1], imgstring))

import wx
from PIL import Image

class CUAVImageViewerFrame(wx.Frame):
    """ The main frame of the viewer
    """    
    def __init__(self, state):
        wx.Frame.__init__(self, None, -1, state.title)
        self.state = state
        self.panel = CUAVImageViewerPanel(self, state)
        self.Bind(wx.EVT_IDLE, self.on_idle)

    def on_idle(self, event):
        '''prevent the main loop spinning too fast'''
        import time
        time.sleep(0.1)

class CUAVImageViewerPanel(wx.Panel):
    """ The image panel
    """    
    def __init__(self, parent, state):
        wx.Panel.__init__(self, parent)
        self.scroll = wx.ScrolledWindow(self, -1)
        self.scroll.SetScrollbars(1, 1, 1000, 1000)
        self.state = state
        self.img = None
        self.redraw_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_redraw_timer, self.redraw_timer)        
        self.redraw_timer.Start(200)
        self.create_main_window()

    def create_main_window(self):
        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.mainSizer)
        
    def on_redraw_timer(self, event):
        state = self.state
        while state.child_pipe.poll():
            (w,h,imgstring) = state.child_pipe.recv()
            if self.img is not None:
                self.mainSizer.Remove(self.imageCtrl)
            self.img = wx.EmptyImage(w,h)
            self.img.SetData(imgstring)
            self.imageCtrl = wx.StaticBitmap(self.scroll, wx.ID_ANY, 
                                             wx.BitmapFromImage(self.img))
            self.mainSizer.Add(self.imageCtrl, 0, wx.ALL|wx.CENTER, 5)
            self.mainSizer.Layout()
            self.mainSizer.Fit(self)
            print("loaded image", w, h)
            self.Refresh()
            
    
if __name__ == "__main__":
    # test the graph
    import sys, time
    next = 1
    w = CUAVImageViewer(title='Image Test')
    while w.is_alive():
        file = sys.argv[next]
        print file
        img = Image.open(file)
        w.ShowImage(img)
        time.sleep(2)
        next += 1
        if next == len(sys.argv):
            next = 1
