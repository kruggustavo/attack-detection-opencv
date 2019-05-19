from hog.WeaponsDetection import WeaponsDetection
import cv2
import numpy as np
import queue
import threading
import time



workpath = "/home/usuario/Documentos/attack-detection-opencv"
dataset_path = workpath +"/Imagenes/Dataset/"

video_width = 480
video_height = 320

# Detector de objetos
weapons = WeaponsDetection(dataset_path)

cap = cv2.VideoCapture("/home/usuario/Documentos/attack-detection-opencv/Videos/2.mp4")

while True:
    hasFrame, frame = cap.read()
    if hasFrame:

        frame = cv2.resize(frame, (480, 360))

        predictions, labels = weapons.detect(frame)
        if len(predictions) > 0 :
            #print(str(labels) + " "+ str(predictions))

            for group in predictions:
                for box in group:
                    print(box)
                    (x, y, xb, yb) = box
                    cv2.rectangle(frame, (x, y), (xb, yb), (0, 255, 0), 1)

        cv2.imshow("Camara", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break
