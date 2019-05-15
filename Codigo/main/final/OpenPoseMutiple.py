import cv2
import time
import numpy as np
from random import randint

class OpenPoseMultiple:
    nPoints = 14

    frameWidth = 0
    frameHeight = 0

    POINTS_LABELS = ["c", "n", "hi", "ci", "mi", "hd", "cd", "md", "hpi", "ri", "pi", "hpd", "rd", "pd", "p"]

    def __init__(self, neuralnet=None, protoFile="", weightsFile=""):
        if neuralnet == None:
            self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        else:
            self.net = neuralnet

    def detectHumanPose(self, frame):
        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]

        inHeight = 360
        inWidth = int((inHeight / self.frameHeight) * self.frameWidth)

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        self.net.setInput(inpBlob)
        output = self.net.forward()

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1

        for part in range(self.nPoints):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            keypoints = self.getKeypoints(probMap, threshold)

            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)

        detected_keypoints = self.getPointsByHuman(detected_keypoints)

        return detected_keypoints

    # Obtiene puntos por persona, no por partes humanas
    def getPointsByHuman(self, detected_keypoints):
        humansNumber = len(detected_keypoints[0])

        humansPointsLists = list({} for i in range(humansNumber))

        if humansNumber > 0:
            for i in range(self.nPoints):
                for j in range(len(detected_keypoints[i])):
                    try:
                        humansPointsLists[j][self.POINTS_LABELS[i]] = detected_keypoints[i][j][0:2]
                    except:
                        print("No point for human " + str(j) + " for body part: " + self.POINTS_LABELS[i])

        return humansPointsLists

    def getKeypoints(self, probMap, threshold=0.1):

        mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

        mapMask = np.uint8(mapSmooth>threshold)
        keypoints = []

        #find the blobs
        contours, b = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints