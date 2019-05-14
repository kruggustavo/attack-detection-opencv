import cv2
import time
import numpy as np

# Codigo basado en https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

class OpenPose:
    MODE = "MPI"
    nPoints = 15
    inWidth = inHeight = 245
    threshold = 0.1

    def __init__(self, neuralnet=None, protoFile="", weightsFile=""):
        if neuralnet == None:
            self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        else:
            self.net = neuralnet


    def detectHumanPose(self, frame):
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)
        output = self.net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > self.threshold:
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)
        return points