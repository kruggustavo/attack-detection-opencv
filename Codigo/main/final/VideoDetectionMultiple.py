from final.NeuralNetwork import NeuralNetwork
from final.OpenPose import OpenPose
from final.Drawer import Drawer
from pygame import mixer

import queue
import threading
import cv2
import numpy as np
import imutils
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
videoFile = workpath +"/Videos/2.mp4"
cap = cv2.VideoCapture(videoFile)
video_width = 420
video_height = 300

# Se procesará cada X frames
process_each_frame = 1
frame_count = process_each_frame

# Dibujador y extractor de angulos
drawer = Drawer()

# Red neuronal de angulos
EPOCHS = 30
nnet = NeuralNetwork("trainingangles.csv", 8)
nnet.trainNetwork(EPOCHS)

max_num_threads = 6
threads_array = []
humansFramesQueue = queue.Queue()               # Frames de personas para procesar
pointsQueue = queue.Queue()                     # Puntos detectados a mostrar
framesQueue = queue.Queue()                          # Frames completos de video
rectsQueue = queue.Queue()                      # Rectangulos de personas detectadas en humanFrames

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
op = OpenPose(protoFile="pose/mpi/pose_deploy_linevec_faster_2_stages.prototxt",
              weightsFile="pose/mpi/pose_iter_150000.caffemodel")

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Hilos consumidores para pose de humano
def consumer():
    global busy, op

    while True:
        frame = humansFramesQueue.get()
        if frame is None:
            return
        points = op.detectHumanPose(frame)
        pointsQueue.put(points)

        if humansFramesQueue.qsize() == 0:
            busy = False

# Hilo consumidor para deteccion de humanos
def multiscaleHumanConsumer():
    while True:
        frame = framesQueue.get()
        if frame is None:
            return
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        rectsQueue.put(rects)

# Emite alerta
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

# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


print(str(max_num_threads) + " consumers threads waiting...")
#Create X threads
for i in range(max_num_threads):
    thread = threading.Thread(target=consumer)
    thread.start()
    threads_array.append(thread)


human = threading.Thread(target=multiscaleHumanConsumer)
human.start()

rects = None
while True:
    hasFrame, frame = cap.read()
    #frame = cv2.imread(img_file)
    if hasFrame:

        frame = cv2.resize(frame, (video_width, video_height))

        frame_count = frame_count - 1
        if frame_count == 0:
            frame_count = process_each_frame

            if framesQueue.qsize() == 0:
                framesQueue.put(frame)

            if rectsQueue.qsize() > 0:
                rects = rectsQueue.get()
                # apply non-maxima suppression to the bounding boxes using a
                # fairly large overlap threshold to try to maintain overlapping
                # boxes that are still people
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                rects = non_max_suppression(rects, overlapThresh=0.65)

                try:
                    if busy == False:
                        if len(rects) > 0:
                            # draw the final bounding boxes
                            for (xA, yA, xB, yB) in rects:
                                crop_img = frame[yA:yB, xA:xB]
                                # A la cola de frames para procesar
                                humansFramesQueue.put(crop_img)
                            busy = True
                except:
                    print("No boxes found yet")


            if pointsQueue.qsize() > 0:

                points = pointsQueue.get()
                #skeletonFrame = drawer.drawSkeletonPoints(emptyFrame.copy(), points, margin=15)

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

                    if netOutput == ATTACK:
                        playWarningMsg()

                    skeletonFrame = drawer.drawSkeletonLines(emptyFrame.copy(), lines, margin=50)

                    cv2.putText(skeletonFrame, ATTACK_STATE[netOutput],
                                (int((video_width / 2) - (len(ATTACK_STATE[val]) * 5)), 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


        frame = np.concatenate((frame, skeletonFrame), axis=1)

        cv2.imshow("Camara", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")