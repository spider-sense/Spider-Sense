# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 02:21:50 2021

@author: derph
"""

from pyblur import PsfBlur_random#LinearMotionBlur_random
import cv2
from PIL import Image

name = "girl_ivan"
img = cv2.cvtColor(cv2.imread("./images/" + name +".jpg"), cv2.COLOR_BGR2GRAY)
blurred = PsfBlur_random(img)
#blurred = LinearMotionBlur_random(img)
blurred.save("./images/" + name + "Blur.jpg")