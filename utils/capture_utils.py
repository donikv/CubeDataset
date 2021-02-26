import cv2
import os
import numpy as np
import time
# if os.name != 'nt':
#     import gphoto2 as gp


def show_full_screen(image, path, duration=3):
    image_path = os.path.join(path, image)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image)
    image = np.dstack((b, g, r))

    window = "window"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, image)

    return window


def capture_from_camera(filename):
    camera = gp.Camera()
    context = gp.Context()
    context.camera_autodetect()
    camera.init(context)
    text = camera.get_summary(context)
    path = camera.capture(gp.GP_CAPTURE_IMAGE, context)
    file = camera.file_get(path.folder, path.name, gp.GP_FILE_TYPE_NORMAL, context)
    file.save(filename)
    buf = file.get_data_and_size()
    camera.exit()
