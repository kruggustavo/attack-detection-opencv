from final.NeuralNetwork import NeuralNetwork
from final.OpenPose import OpenPose
from final.Drawer import Drawer
from pygame import mixer

import queue
import threading
import cv2
import numpy as np
import time

# Attack detection
# Detector de ataques y amenazas humanas en imagenes secuenciales mediante el entrenamiento de redes neuronales
# Trabajo final de Grado para la obtencion del titulo de Ingeniero Informatico
# Autor: Gustavo Krug
# Universidad Catolica Nuestra Señora de la Asuncion, 2019
#

print("Human attack detection")

# Parametros generales
ATTACK = 1
NO_ATTACK = 0

ATTACK_STATE = {}
ATTACK_STATE[NO_ATTACK] = "Pose no compatible con ataque"
ATTACK_STATE[ATTACK] = "Pose compatible con ataque!"

workpath = "/home/usuario/Documentos/attack-detection-opencv"
videoFile = workpath +"/Videos/1.mp4"
cap = cv2.VideoCapture(videoFile)
video_width = 420
video_height = 300


# Dibujador y extractor de angulos
drawer = Drawer()

# Red neuronal de angulos
EPOCHS = 2
nnet = NeuralNetwork("trainingangles.csv", 8)
nnet.trainNetwork(EPOCHS)

max_num_threads = 4
threads_array = []
framesQueue = queue.Queue()                     # Frames para procesar
pointsQueue = queue.Queue()                     # Puntos detectados a mostrar
busy = False                                    # Flag for empty queue

val = 0                                         # 1 es agresion
Xseconds = 10                                   # Cantidad de segundos que deben transcurrir para repetir el mensaje de agresion

emptyFrame = np.zeros((video_height, video_width, 3), np.uint8)
skeletonFrame = emptyFrame

mixer.init()
mixer.music.load("47528846.mp3")
agresionTime = time.time() - Xseconds

hasFrame = True
lastVal = NO_ATTACK

# Red neuronal de poses
op = OpenPose(protoFile="pose/mpi/pose_deploy_linevec_faster_1_stages.prototxt",
              weightsFile="pose/mpi/pose_iter_150000.caffemodel")

def consumer():
    global busy, op

    while True:
        frame = framesQueue.get()
        if frame is None:
            return
        busy = True
        points = op.detectHumanPose(frame)
        pointsQueue.put(points)
        busy = False


def playWarningMsg():
    global lastVal, agresionTime, mixer
    # Si han pasado segundos desde el ultimo ataque, habilitar alertas
    if lastVal == ATTACK and (time.time() - agresionTime) > Xseconds:
        lastVal = val

    # Si agresion existe luego de X segundos emitir audio
    if (time.time() - agresionTime) > Xseconds:
        mixer.music.play()
        agresionTime = time.time()
        lastVal = val

    # Alerta visual durante la mitad del tiempo de ataque
    if lastVal == ATTACK and (time.time() - agresionTime) < (Xseconds * 0.5):
        cv2.circle(skeletonFrame, (video_width - 30, video_height - 30), 30, (0, 0, 255), -1)


print(str(max_num_threads) + " consumers threads waiting...")
#Create X threads
for i in range(max_num_threads):
    thread = threading.Thread(target=consumer)
    thread.start()
    threads_array.append(thread)

print("Stand by for threads to start")
time.sleep(Xseconds * max_num_threads)

print("Threads loaded...")
while True:
    hasFrame, frame = cap.read()
    #frame = cv2.imread(img_file)
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

                angles = np.array([list(angles.values())])

                if len(angles) > 0:
                    print(angles)

                try:
                    netOutput = int(nnet.predict(angles))
                except:
                    netOutput = NO_ATTACK

                cv2.putText(skeletonFrame, ATTACK_STATE[netOutput],
                            (int((video_width / 2) - (len(ATTACK_STATE[val]) * 5)), 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

                if netOutput == ATTACK:
                    playWarningMsg()

                # Dibujamos tronco
                if "trunkPoints" in lines:
                    pointA = lines["trunkPoints"][0]
                    pointB = lines["trunkPoints"][1]
                    cv2.line(skeletonFrame, pointA, pointB, (100, 7, 65), 3, lineType=cv2.LINE_AA)

        frame = np.concatenate((frame, skeletonFrame), axis=1)


        cv2.imshow("Camara", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")