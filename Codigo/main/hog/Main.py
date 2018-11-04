from hog.BodyPartsDetection import BodyPartsDetection
import cv2

#
# Referencia: http://www.hackevolve.com/create-your-own-object-detector/
#

workpath = "/home/gustavokrug/Documents/attack-detection-opencv"
video_file = workpath +"/Videos/3.mp4"
dataset_path = workpath +"/Imagenes/Dataset/"
properties = workpath + "/Imagenes/Properties/"

bodyDetector = BodyPartsDetection(dataset_path, properties)

video_width = 480
video_height = 320
cap = cv2.VideoCapture(video_file)
detected_count = 0

while True:
    r, frame = cap.read()

    if r:
        frame = cv2.resize(frame, (video_width, video_height))  # Downscale to improve frame rate
        predictions, labels = bodyDetector.detect(frame)

        detected_count = detected_count + len(predictions)

        if len(predictions) > 0:
            humanBodies = bodyDetector.determineHumanBody(predictions, labels)
            if len(humanBodies) > 0:
                print("Detected full bodies " + str(len(humanBodies)))
                xa, ya, x2a, y2a = humanBodies[0][0][0]                     # arms box
                xb, yb, x2b, y2b = humanBodies[0][1][0]                     # legs box
                cv2.rectangle(frame, (xa, ya), (x2b, y2b), (0, 0, 255), 2)  # rectangle for full body

        cv2.putText(frame, str(detected_count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Detected", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")