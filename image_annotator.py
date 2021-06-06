# importing the module
import cv2
import utils.file_utils as fu

xs, ys = 0,0
is_zooming_in = False
dx, dy = 1, 1
imgx, imgy = 0,0

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    global xs, ys, dx, dy
    global is_zooming_in

    if event == cv2.EVENT_LBUTTONDOWN and flags == cv2.EVENT_FLAG_CTRLKEY:
        if xs != 0 or ys != 0:
            xs = 0
            ys = 0
            return
        is_zooming_in = True
        xs, ys = x, y
    elif event == cv2.EVENT_LBUTTONUP and is_zooming_in:
        

    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 255), 2)
        cv2.imshow('image', img)


# driver function
if __name__ == "__main__":
    # reading the image
    img = fu.load_cr2('/Volumes/Jolteon/fax/raws1/1.NEF', pp=True)
    imgx, imgy = img.shape[1], img.shape[0]

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()