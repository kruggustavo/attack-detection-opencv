from hog.GatherAnnotations import Annotations
from hog.ObjectDetector import ObjectDetector
import cv2

#
# Referencia: http://www.hackevolve.com/create-your-own-object-detector/
#

workpath = "/home/gustavokrug/Documents/attack-detection-opencv"
video_file = workpath +"/Videos/3.mp4"
dataset_path = workpath +"/Imagenes/Dataset/arms_front--walking"

print("Getting annotations... " + dataset_path)
annotationsObj = Annotations(dataset_path=dataset_path)
annotations, imagePaths, label = annotationsObj.annotate()

print("Get Hog descriptors from train images in " + dataset_path)
detector = ObjectDetector()
detector.hog_descriptors(imagePaths, annotations, visualizeHog=False)

print("Detect in video with SVM classifier")
# 800 x 600
video_width = 480
video_height = 320
cap = cv2.VideoCapture(video_file)
detected_count = 0

while True:
    r, frame = cap.read()

    if r:
        frame = cv2.resize(frame, (video_width, video_height))  # Downscale to improve frame rate


        frame, preds = detector.detect(frame)
        detected_count = detected_count + len(preds)

        cv2.putText(frame, str(detected_count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Detected", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break



print("end")