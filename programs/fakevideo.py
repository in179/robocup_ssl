import cv2
import numpy as np
import pyvirtualcam
from time import sleep

img = cv2.imread('output2.png')
h, w, _ = img.shape

with pyvirtualcam.Camera(width=w, height=h, fps=30) as cam:
    while True:
        cam.send(img)
        cam.sleep_until_next_frame()
