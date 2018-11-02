import cv2
import time
import numpy as np

# video size
frameWidth = 320
frameHeight = 240

workpath = "/home/gustavokrug/Documents/Tesis/"
video_file = workpath +"/Videos/2.mp4"

# Specify the paths for the 2 files
protoFile = workpath + "/Desarrollo/coco/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = workpath + "/Desarrollo/coco/pose_iter_160000.caffemodel"
#frame = cv2.imread(workpath + "/Imagenes/tst.jpg")


MODE = "MPI"

nPoints = 15
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
              [11, 12], [12, 13]]

inWidth = 200
inHeight = 200
threshold = 0.1

input_source = video_file
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

process_each_frame = 10                         # Se procesará cada X frames
frame_count = process_each_frame

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()

    if not hasFrame:
        cv2.waitKey()
        break

    frame_count = frame_count - 1
    if frame_count == 0:
        frame_count = process_each_frame

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        color = 0
        for pair in POSE_PAIRS:
            color = color + 5
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, color, 255), 1, lineType=cv2.LINE_AA)


        # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow('Output-Keypoints', frameCopy)
        cv2.imshow('Output-Skeleton', frame)
