from hog.BodyPartsDetection import BodyPartsDetection
import cv2
import numpy as np

#
# Referencia: http://www.hackevolve.com/create-your-own-object-detector/
#

workpath = "/home/gustavokrug/Documents/attack-detection-opencv"

video_file = workpath +"/Videos/3.mp4"
dataset_path = workpath +"/Imagenes/Dataset/"
properties = workpath + "/Imagenes/Properties/"
keyPoints = workpath + "/Imagenes/KeyPoints/"

bodyDetector = BodyPartsDetection(dataset_path, properties, keyPoints)

video_width = 480
video_height = 320
cap = cv2.VideoCapture(video_file)
detected_count = 0

# Referencias
# c = cabeza, h = hombro (i, d), c = codo (i, d), m = mano (i, d), p = pecho
# h = cadera, r = rodilla(i, d), p = pie (i, d)
POSE_PAIRS = [["c", "p"], ["hd", "hi"], ["hi", "ci"], ["ci", "mi"], ["hd", "cd"], ["cd", "md"], ["p", "h"], ["h", "ri"], ["ri", "pi"], ["h", "rd"], ["rd", "pd"]]

def drawRectangleOnFrame(frame, humanBodies):
    if len(humanBodies) > 0:
        xa, ya, x2a, y2a = humanBodies[0][0][0]  # arms box
        xb, yb, x2b, y2b = humanBodies[0][1][0]  # legs box
        cv2.rectangle(frame, (xa, ya), (x2b, y2b), (0, 0, 255), 2)  # rectangle for full body

def drawHumanPose(posekeypoints):
    if len(posekeypoints) > 0:
        poseFrame = np.zeros((320, 260, 3), np.uint8)

        height, width = poseFrame.shape[:2]
        origins = {8: 0, 5: int(height / 2) - 10}           # 8 points for Arms, start at 0
                                                            # 5 points for legs, start after middle frame
        pointsMap = {}
        for bodyparts in posekeypoints:
            yOrigin = origins[len(bodyparts)]
            for point in bodyparts:
                for id in point :                           # Point identificator: mi, md, c, ri, ...
                    x = int(point[id]["x"])
                    y = int(point[id]["y"])
                    y = int(y / 2) + yOrigin

                    pointsMap[id] = (x, y)                  # Points map to draw lines

                    cv2.circle(poseFrame, (x, y), 3, (0, 0, 255), -1)
                    cv2.putText(poseFrame, id, (x + 10 , y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for pair in POSE_PAIRS:
            try:
                cv2.line(poseFrame, pointsMap[pair[0]], pointsMap[pair[1]], (0, 150, 25), 1, lineType=cv2.LINE_AA)
            except:
                pass

        cv2.imshow("Pose", poseFrame)


while True:
    r, frame = cap.read()

    if r:
        frame = cv2.resize(frame, (video_width, video_height))  # Downscale to improve frame rate
        predictions, labels, lblKeyPoints = bodyDetector.detect(frame)

        detected_count = detected_count + len(predictions)

        if len(predictions) > 0:
            humanBodies = bodyDetector.determineHumanBody(predictions, labels)
            drawRectangleOnFrame(frame, humanBodies)
            drawHumanPose(lblKeyPoints)

        cv2.putText(frame, str(detected_count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Detected", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")