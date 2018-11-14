from hog.BodyPartsDetection import BodyPartsDetection
import cv2
import numpy as np
import queue
import threading
import time

#
# Referencia: http://www.hackevolve.com/create-your-own-object-detector/
#

workpath = "/home/gustavokrug/Documents/attack-detection-opencv"
video_file = workpath +"/Videos/3.mp4"
dataset_path = workpath +"/Imagenes/Dataset/"
properties = workpath + "/Imagenes/Properties/"
keyPoints = workpath + "/Imagenes/KeyPoints/"

video_width = 480
video_height = 320
cap = cv2.VideoCapture(video_file)

# Referencias
# c = cabeza, h = hombro (i, d), c = codo (i, d), m = mano (i, d), p = pecho
# h = cadera, r = rodilla(i, d), p = pie (i, d)
POSE_PAIRS = [["c", "p"], ["hd", "hi"], ["hi", "ci"], ["ci", "mi"], ["hd", "cd"], ["cd", "md"], ["p", "h"], ["h", "ri"], ["ri", "pi"], ["h", "rd"], ["rd", "pd"]]

process_each_frame = 2                         # Se procesará cada X frames
frame_count = process_each_frame

framesQueue = queue.Queue()
skeletonQueue = queue.Queue()

max_num_threads = 4
threads_array = []

bodyDetector = BodyPartsDetection(dataset_path, properties, keyPoints)
detect_cnt = 0

blankFrame = np.zeros((320, 260, 3), np.uint8)

def consumerDetectorThread():
    global detect_cnt, bodyDetector
    print("Consumer waiting...")
    while True:
        frame = framesQueue.get()
        if frame is None:
            return

        predictions, labels, lblKeyPoints = bodyDetector.detect(frame)

        detect_cnt = detect_cnt + len(predictions)

        if len(predictions) > 0:
            print(str(time.time())[-6:] + ' Half/full Parts detected: ' + str(len(lblKeyPoints)))
            #humanBodies = bodyDetector.determineHumanBody(predictions, labels)
            #if len(humanBodies) > 0:
            #    xa, ya, x2a, y2a = humanBodies[0][0][0]  # arms box
            #    xb, yb, x2b, y2b = humanBodies[0][1][0]  # legs box
            #    cv2.rectangle(frame, (xa, ya), (x2b, y2b), (0, 0, 255), 2)  # rectangle for full body
            targetFrame = drawHumanPose(lblKeyPoints, blankFrame.copy())
            skeletonQueue.put(targetFrame)

def drawHumanPose(posekeypoints, targetFrame):
    if len(posekeypoints) > 0:
        height, width = targetFrame.shape[:2]
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

                    cv2.circle(targetFrame, (x, y), 3, (0, 0, 255), -1)
                    cv2.putText(targetFrame, id, (x + 10 , y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for pair in POSE_PAIRS:
            try:
                cv2.line(targetFrame, pointsMap[pair[0]], pointsMap[pair[1]], (0, 150, 25), 1, lineType=cv2.LINE_AA)
            except:
                pass

        return targetFrame


#Create X threads
for i in range(max_num_threads):
    thread = threading.Thread(target=consumerDetectorThread)
    thread.start()
    threads_array.append(thread)

while True:
    r, frame = cap.read()

    if r:
        frame_count = frame_count - 1
        if frame_count == 0:
            frame_count = process_each_frame

            frame = cv2.resize(frame, (video_width, video_height))
            framesQueue.put(frame)

            if skeletonQueue.qsize() > 0:
                sk = skeletonQueue.get()
                cv2.imshow("skeleton", sk)

            cv2.imshow("Detected", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")