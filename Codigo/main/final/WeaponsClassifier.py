import numpy as np
import cv2
import imutils
import datetime

class WeaponsClassifier:

    min = (40, 40)

    def __init__(self, cascadeFile="", minSize=(100, 100)):
        self.min = minSize
        self.cascade = cv2.CascadeClassifier(cascadeFile)


    def classifyOneToOthers(self, frame):
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        return self.cascade.detectMultiScale(gray, 1.5, 5, minSize=min)
