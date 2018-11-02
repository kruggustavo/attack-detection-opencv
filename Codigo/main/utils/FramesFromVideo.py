import cv2
import numpy as np
from utils.BoxSelector import BoxSelector
from utils.MyImageUtils import MyImageUtils

ARMS = 0
LEGS = 1

def getPortionOfBoddy(mode=-1, x=0, y=0, h=0, w=0):
    if mode == ARMS:
        return y,y+int(h/2),x,x+w
    elif mode == LEGS:
        return y+int(h/2),y+h,x,x+w
    else:
        return y,y+h,x,x+w


myImageUtils = MyImageUtils()
workpath = "D:/Google Drive/Tesis"
video_file = workpath +"/Videos/3.mp4"

video_width = 800
video_height = 600

sample_width = 520
sample_height = 980

cutted_images_folder = workpath + "/Imagenes/dataset/"

video_area = video_width * video_height     # Area en pixeles del tamaño del frame

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(video_file)
humans_count = 0
margins = 10
admited_percentage_image_size = 8              # Porcentaje de tamaño de imagenes positivas
process_each_frame = 1                         # Se procesará cada X frames
frame_count = process_each_frame
humans_count = 0

while True:
    r, frame = cap.read()
    roi = None
    if r:
        frame_count = frame_count - 1
        if frame_count == 0:
            frame_count = process_each_frame

            frame = cv2.resize(frame, (video_width, video_height))  # Downscale to improve frame rate
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            rects1, weights1 = hog.detectMultiScale(frame)

            for (x, y, w, h) in rects1:
                x = x - 10
                w = w + 10
                y = y + 10
                percentage = (w * h * 100) / (video_area)
                if percentage > admited_percentage_image_size:
                    y, h, x, w = getPortionOfBoddy(LEGS, x, y, h, w)
                    print(y,h,x,w)
                    roi = frame[y:h, x:w]
                    r = 600.0 / roi.shape[1]
                    dim = (600, int(int(roi.shape[0] * r) / 10) * 10)
                    roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)

                    outputfilename = cutted_images_folder + ("frame%d.jpg" % humans_count)
                    humans_count += 1
                    print("Humanos detectados :" + str(humans_count))
                    cv2.imwrite(outputfilename, roi)

            #cv2.imshow("frame", frame)

            #k = cv2.waitKey(1)
            #if k & 0xFF == ord("q"):  # Exit condition
            #    break

