from final.OpenPose import OpenPose
from final.Drawer import Drawer

import queue
import threading
import cv2
import numpy as np
#
# Pose generation
# Detector de ataques y amenazas humanas en imagenes secuenciales mediante el entrenamiento de redes neuronales
# Trabajo final de Grado para la obtencion del titulo de Ingeniero Informatico
# Autor: Gustavo Krug
# Universidad Catolica Nuestra Señora de la Asuncion, 2019
#

print("Human pose generation from video")

# Parametros generales
ATTACK_POSE = 1
NO_ATTACK_POSE = 0

POSE = ATTACK_POSE

workpath = "/home/usuario/Documentos/attack-detection-opencv"
cap = cv2.VideoCapture(0)
video_width = 480
video_height = 320

# Red neuronal de poses
op = OpenPose(protoFile="pose/mpi/pose_deploy_linevec_faster_2_stages.prototxt", weightsFile="pose/mpi/pose_iter_150000.caffemodel")

# Dibujador y extractor de angulos
drawer = Drawer()

max_num_threads = 3
threads_array = []
framesQueue = queue.Queue()                     # Frames para procesar
pointsQueue = queue.Queue()                     # Puntos detectados a mostrar
busy = False                                    # Flag for empty queue

emptyFrame = np.zeros((video_height, video_width, 3), np.uint8)
skeletonFrame = emptyFrame

def consumer():
    global busy, op, emptyFrame
    while True:
        frame = framesQueue.get()
        if frame is None:
            return
        busy = True
        points = op.detectHumanPose(frame)
        pointsQueue.put(points)
        busy = False


print(str(max_num_threads) + " consumers threads waiting...")
#Create X threads
for i in range(max_num_threads):
    thread = threading.Thread(target=consumer)
    thread.start()
    threads_array.append(thread)


hasFrame = True

while True:
    hasFrame, frame = cap.read()
    if hasFrame:

        frame = cv2.resize(frame, (video_width, video_height))

        if busy == False:
            framesQueue.put(frame)

        if pointsQueue.qsize() > 0:
            points = pointsQueue.get()
            skeletonFrame = drawer.drawSkeleton(emptyFrame.copy(), points)

            if len(points) > 0:
                labeledPoints = drawer.getLabeledPoints(points)
                angles, lines = drawer.getBodyAngles(labeledPoints)

                angles["pose"] = POSE

                angles = np.array([list(angles.values())])
                anglesStr = str(angles)
                if len(angles) > 0 and "180 180 180 180 180 180 180 180" not in anglesStr:
                    anglesStr = anglesStr.replace(". ]]", "]]")
                    anglesStr = anglesStr.replace(". ", ".0")
                    anglesStr = anglesStr.replace(" ]]", "]]").replace("[[ ", "[[")
                    anglesStr = anglesStr.replace(" ]]", "]]").replace("[[ ", "[[")
                    anglesStr = anglesStr.replace("  ", " ").replace("  ", " ")
                    anglesStr = anglesStr.replace(" ", ",")
                    anglesStr = anglesStr.replace("[", "").replace("]", "")
                    print(anglesStr)

                # Dibujamos tronco
                if "trunkPoints" in lines:
                    pointA = lines["trunkPoints"][0]
                    pointB = lines["trunkPoints"][1]
                    cv2.line(skeletonFrame, pointA, pointB, (100, 7, 65), 2, lineType=cv2.LINE_AA)

        frame = np.concatenate((frame, skeletonFrame), axis=1)


        cv2.imshow("Camara", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")