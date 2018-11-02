import cv2
import time
import os
import numpy as np


full_body_cascade = cv2.CascadeClassifier("C:/Users/Gustavo Krug/Desktop/Tesis/Desarrollo/cascade/haarcascade_fullbody.xml")
upper_body_cascade = cv2.CascadeClassifier("C:/Users/Gustavo Krug/Desktop/Tesis/Desarrollo/cascade/haarcascade_upperbody.xml")

cap = cv2.VideoCapture("C:/Users/Gustavo Krug/Desktop/Tesis/Videos/1.mp4")

video_width = 360
video_height = 280
video_area = video_width * video_height     # Area en pixeles del tamaño del frame

humans_detected = 0

admited_percentage_image_size = 10           # Porcentaje de tamaño de imagenes positivas

while True:
    r, frame = cap.read()

    if r:
        frame = cv2.resize(frame, (video_width, video_height))  # Downscale to improve frame rate

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Haar-cascade classifier needs a grayscale image

        rects4 = full_body_cascade.detectMultiScale(gray_frame)

        for (x, y, w, h) in rects4:
            percentage = (w * h * 100) / (video_area)
            if percentage > admited_percentage_image_size:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                humans_detected += 1
                print("Cascade detections : ", humans_detected)

        cv2.putText(frame, ("Cascade detections : " + str(humans_detected)), (1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Cascade", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break