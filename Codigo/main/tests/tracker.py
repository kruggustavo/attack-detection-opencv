import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("D:/Google Drive/Tesis/Videos/2.mp4")

video_width = 320
video_height = 240
margins = 10
video_area = video_width * video_height     # Area en pixeles del tamaño del frame

humans_detected = 0
last_humans_detected = 0

admited_percentage_image_size = 10           # Porcentaje de tamaño de imagenes positivas



while True:
    r, frame = cap.read()

    if r:
        frame = cv2.resize(frame, (video_width, video_height))  # Downscale to improve frame rate

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        rects1, weights1 = hog.detectMultiScale(gray_frame)

        for (x, y, w, h) in rects1:
            percentage = (w * h * 100) / (video_area)
            if percentage > admited_percentage_image_size:
                if x > margins:
                    x = x - margins
                if x + w < video_width - margins:
                    w = w + margins
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)



        cv2.imshow("Hog", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break