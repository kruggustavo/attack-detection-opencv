import numpy as np
import cv2

workpath = "D:/Google Drive/Tesis"
video_file = workpath +"/Videos/2.mp4"
cap = cv2.VideoCapture(video_file)
video_width = 480
video_height = 320


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


subtractor = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=5000, detectShadows=False)
r, first_frame = cap.read()
first_frame = cv2.resize(first_frame, (video_width, video_height))



while True:
    r, frame = cap.read()
    frame = cv2.resize(frame, (video_width, video_height))

    mask = subtractor.apply(frame)
    mask = cv2.merge((mask, mask, mask))

    difference = cv2.absdiff(first_frame, frame)
    difference = 255 - difference

    mixed = cv2.addWeighted(frame, 1.5, mask, -0.5, 10)


    #rects1, weights1 = hog.detectMultiScale(difference)
    cv2.imshow("diff", difference)
    cv2.imshow("mask", mask)
    cv2.imshow("mixed", mixed)



    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()