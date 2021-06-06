# -*- coding: utf-8 -*-
import cv2
import numpy as np
import utils.file_utils as fu
import os

LF = '\n'
CRLF = '\r\n'
CR = '\r'

class PanZoomWindow(object):
    """ Controls an OpenCV window. Registers a mouse listener so that:
        1. right-dragging up/down zooms in/out
        2. right-clicking re-centers
        3. trackbars scroll vertically and horizontally
    You can open multiple windows at once if you specify different window names.
    You can pass in an onLeftClickFunction, and when the user left-clicks, this
    will call onLeftClickFunction(y,x), with y,x in original image coordinates."""
    def __init__(self, img, windowName = 'PanZoomWindow', onLeftClickFunction = None):
        self.WINDOW_NAME = windowName
        self.H_TRACKBAR_NAME = 'x'
        self.V_TRACKBAR_NAME = 'y'
        self.img = img
        self.onLeftClickFunction = onLeftClickFunction
        self.onLeftShiftClickFunction = None
        self.TRACKBAR_TICKS = 1000
        self.panAndZoomState = PanAndZoomState(img.shape, self)
        self.lButtonDownLoc = None
        self.mButtonDownLoc = None
        self.rButtonDownLoc = None
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.redrawImage()
        cv2.setMouseCallback(self.WINDOW_NAME, self.onMouse)
        cv2.createTrackbar(self.H_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onHTrackbarMove)
        cv2.createTrackbar(self.V_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onVTrackbarMove)
    def onMouse(self,event, x,y,_ignore1,_ignore2):
        """ Responds to mouse events within the window.
        The x and y are pixel coordinates in the image currently being displayed.
        If the user has zoomed in, the image being displayed is a sub-region, so you'll need to
        add self.panAndZoomState.ul to get the coordinates in the full image."""
        if event == cv2.EVENT_MOUSEMOVE:
            return
        elif event == cv2.EVENT_RBUTTONDOWN:
            #record where the user started to right-drag
            self.mButtonDownLoc = np.array([y,x])
        elif event == cv2.EVENT_RBUTTONUP and self.mButtonDownLoc is not None:
            #the user just finished right-dragging
            dy = y - self.mButtonDownLoc[0]
            pixelsPerDoubling = 0.2*self.panAndZoomState.shape[0] #lower = zoom more
            changeFactor = (1.0+abs(dy)/pixelsPerDoubling)
            changeFactor = min(max(1.0,changeFactor),5.0)
            if changeFactor < 1.05:
                dy = 0 #this was a click, not a draw. So don't zoom, just re-center.
            if dy > 0: #moved down, so zoom out.
                zoomInFactor = 1.0/changeFactor
            else:
                zoomInFactor = changeFactor
#            print("zoomFactor: %s"%zoomFactor)
            self.panAndZoomState.zoom(self.mButtonDownLoc[0], self.mButtonDownLoc[1], zoomInFactor)
        elif event == cv2.EVENT_LBUTTONDOWN:
            #the user pressed the left button.
            coordsInDisplayedImage = np.array([y,x])
            if np.any(coordsInDisplayedImage < 0) or np.any(coordsInDisplayedImage > self.panAndZoomState.shape[:2]):
                print("you clicked outside the image area")
            else:
                print("you clicked on %s within the zoomed rectangle"%coordsInDisplayedImage)
                coordsInFullImage = self.panAndZoomState.ul + coordsInDisplayedImage
                print("this is %s in the actual image"%coordsInFullImage)
                if self.onLeftClickFunction is not None and _ignore1 - 1 == cv2.EVENT_FLAG_CTRLKEY:
                    self.onLeftClickFunction(coordsInFullImage[0],coordsInFullImage[1])
                    self.draw_x(coordsInFullImage[1], coordsInFullImage[0])
                if self.onLeftShiftClickFunction is not None and _ignore1 - 1 == cv2.EVENT_FLAG_SHIFTKEY:
                    self.onLeftShiftClickFunction(coordsInFullImage[0],coordsInFullImage[1])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(self.img, "y", (coordsInFullImage[1], coordsInFullImage[0]), font,
                                0.5, color=(0, int(self.img.max()), 0), thickness=2)
                    self.redrawImage()
        #you can handle other mouse click events here
    def onVTrackbarMove(self,tickPosition):
        self.panAndZoomState.setYFractionOffset(float(tickPosition)/self.TRACKBAR_TICKS)
    def onHTrackbarMove(self,tickPosition):
        self.panAndZoomState.setXFractionOffset(float(tickPosition)/self.TRACKBAR_TICKS)
    def draw_x(self, x,y):
        cv2.circle(self.img, (x,y), 5, color=(0, int(self.img.max()), 0), thickness=-1)
        self.redrawImage()
    def redrawImage(self):
        pzs = self.panAndZoomState
        cv2.imshow(self.WINDOW_NAME, self.img[pzs.ul[0]:pzs.ul[0]+pzs.shape[0], pzs.ul[1]:pzs.ul[1]+pzs.shape[1]])

class PanAndZoomState(object):
    """ Tracks the currently-shown rectangle of the image.
    Does the math to adjust this rectangle to pan and zoom."""
    MIN_SHAPE = np.array([50,50])
    def __init__(self, imShape, parentWindow):
        self.ul = np.array([0,0]) #upper left of the zoomed rectangle (expressed as y,x)
        self.imShape = np.array(imShape[0:2])
        self.shape = self.imShape #current dimensions of rectangle
        self.parentWindow = parentWindow
    def zoom(self,relativeCy,relativeCx,zoomInFactor):
        self.shape = (self.shape.astype(np.float)/zoomInFactor).astype(np.int)
        #expands the view to a square shape if possible. (I don't know how to get the actual window aspect ratio)
        self.shape[:] = np.max(self.shape)
        self.shape = np.maximum(PanAndZoomState.MIN_SHAPE,self.shape) #prevent zooming in too far
        c = self.ul+np.array([relativeCy,relativeCx])
        self.ul = (c-self.shape/2).astype(np.int)
        self._fixBoundsAndDraw()
    def _fixBoundsAndDraw(self):
        """ Ensures we didn't scroll/zoom outside the image.
        Then draws the currently-shown rectangle of the image."""
#        print("in self.ul: %s shape: %s"%(self.ul,self.shape))
        self.ul = np.maximum(0,np.minimum(self.ul, self.imShape-self.shape))
        self.shape = np.minimum(np.maximum(PanAndZoomState.MIN_SHAPE,self.shape), self.imShape-self.ul)
#        print("out self.ul: %s shape: %s"%(self.ul,self.shape))
        yFraction = float(self.ul[0])/max(1,self.imShape[0]-self.shape[0])
        xFraction = float(self.ul[1])/max(1,self.imShape[1]-self.shape[1])
        cv2.setTrackbarPos(self.parentWindow.H_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,int(xFraction*self.parentWindow.TRACKBAR_TICKS))
        cv2.setTrackbarPos(self.parentWindow.V_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,int(yFraction*self.parentWindow.TRACKBAR_TICKS))
        self.parentWindow.redrawImage()
    def setYAbsoluteOffset(self,yPixel):
        self.ul[0] = min(max(0,yPixel), self.imShape[0]-self.shape[0])
        self._fixBoundsAndDraw()
    def setXAbsoluteOffset(self,xPixel):
        self.ul[1] = min(max(0,xPixel), self.imShape[1]-self.shape[1])
        self._fixBoundsAndDraw()
    def setYFractionOffset(self,fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way down the image."""
        self.ul[0] = int(round((self.imShape[0]-self.shape[0])*fraction))
        self._fixBoundsAndDraw()
    def setXFractionOffset(self,fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way right on the image."""
        self.ul[1] = int(round((self.imShape[1]-self.shape[1])*fraction))
        self._fixBoundsAndDraw()

class CoordinatesWriter(object):

    def __init__(self, directory, file):
        self.d = directory
        os.makedirs(directory, exist_ok=True)
        self.f = open(os.path.join(directory, file), 'a')

    def write_coords(self, y, x):
        self.f.write(f"{x} {y} ")
        self.f.flush()

    def write_coords_func(self):
        def w(y,x):
            self.write_coords(y,x)
        return w

    def new_line_func(self):
        def w(y, x):
            self.new_line()

        return w

    def new_line(self):
        self.f.write(LF if os.name == "nt" else CRLF)
        self.f.flush()

    def finalize(self):
        self.f.flush()
        self.f.close()

if __name__ == "__main__":
    d = '/Volumes/Jolteon/fax/raws4/nikon'
    # d = 'C:/Users/Donik/Desktop/raws6/iphone2'
    for i in range(12, 200):
        try:
            img_n = d + f'/{i}.NEF'
            img = fu.load_cr2(img_n, pp=True)
        except:
            continue
        writer = CoordinatesWriter(f'{d}', f'{i}.txt')
        window = PanZoomWindow(img, img_n)
        window.onLeftClickFunction = writer.write_coords_func()
        window.onLeftShiftClickFunction = writer.new_line_func()
        key = -1
        while key != ord('q') and key != 27: # 27 = escape key
            #the OpenCV window won't display until you call cv2.waitKey()
            key = cv2.waitKey(5)#User can press 'q' or ESC to exit.
        writer.finalize()
        cv2.destroyAllWindows()
    quit()