import cv2
import os
import numpy as np
import time


def show_full_screen(image, path, duration=3):
    image_path = os.path.join(path, image)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    window = "window"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, image)

    return window


def capture_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        captured, frame = cap.read()
        # cv2.imshow('frame', frame)
        if captured:
            return frame

    # t00 = time.time()
    # while True:
    #     _, frame = cap.read()
    #     cv2.imshow('frame', frame)
    #     return frame
    #     k = cv2.waitKey(5) & 0xFF
    #     t01 = time.time()
    #     print("{} fps, size: {}".format(int(1. / (t01 - t00)), frame.shape))
    #     t00 = time.time()
    #     if k == 27:
    #         break