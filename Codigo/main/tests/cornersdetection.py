import cv2
from utils.MyImageUtils import MyImageUtils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

workpath = "D:/Google Drive/Tesis"
video_file = workpath +"/Videos/2.mp4"

cap = cv2.VideoCapture(video_file)

video_width = 360
video_height = 280
video_area = video_width * video_height     # Area en pixeles del tamaño del frame

admited_percentage_image_size = 5           # Porcentaje de tamaño de imagenes positivas

myImageUtils = MyImageUtils()

while True:
    r, frame = cap.read()

    if r:
        #frame = cv2.resize(frame, (video_width, video_height))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        rects1, weights1 = hog.detectMultiScale(gray_frame)

        for (x, y, w, h) in rects1:
            percentage = (w * h * 100) / (video_area)
            if percentage > admited_percentage_image_size:
                subImage = frame[y:y + h, x:x + w]
                subImage = myImageUtils.get_image_with_corners(subImage, 12)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        cv2.imshow("Hog", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break