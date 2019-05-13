from hog.BodyPartsDetection import BodyPartsDetection
from hog.Drawer import Drawer
from hog.NeuralNetwork import NeuralNetwork
import cv2
import numpy as np
import queue
import threading

#
# Attack detection
# Detector de ataques y amenazas humanas en imagenes secuenciales mediante el entrenamiento de redes neuronales
# Trabajo final de Grado para la obtencion del titulo de Ingeniero Informatico
# Autor: Gustavo Krug
# Universidad Catolica Nuestra Señora de la Asuncion, 2019
#

# Parametros generales
ATTACK_STATE = {}
ATTACK_STATE[0] = "No attack detected."
ATTACK_STATE[1] = "Attack detected!"

workpath = "/home/usuario/Documentos/attack-detection-opencv"
video_file = workpath +"/Videos/3.mp4"
dataset_path = workpath +"/Imagenes/Dataset/"
properties = workpath + "/Imagenes/Properties/"
keyPoints = workpath + "/Imagenes/KeyPoints/"

video_width = 480
video_height = 320
cap = cv2.VideoCapture(video_file)

process_each_frame = 2                          # Se procesará cada X frames
frame_count = process_each_frame

# Colas de datos
framesQueue = queue.Queue()                     # Frames para procesar
skeletonQueue = queue.Queue()                   # Frames con puntos del esqueleto ya dibujados
pointsQueue = queue.Queue()                     # Array de puntos del esqueleto detectado

max_num_threads = 3
threads_array = []

# Detector de objetos
bodyDetector = BodyPartsDetection(dataset_path, properties, keyPoints)

# Dibujador y manejador de puntos geometricos de poses
drawer = Drawer()

detect_cnt = 0

# Red neuronal
nnet = NeuralNetwork("trainingangles.csv", 8)
nnet.trainNetwork()

def consumerDetectorThread():
    global detect_cnt, bodyDetector
    while True:
        frame = framesQueue.get()
        if frame is None:
            return

        predictions, labels, lblKeyPoints = bodyDetector.detect(frame)

        detect_cnt = detect_cnt + len(predictions)

        if len(predictions) > 0:
            targetFrame = drawer.drawHumanPose(lblKeyPoints)
            skeletonQueue.put(targetFrame)
            pointsQueue.put(lblKeyPoints)


print(str(max_num_threads) + " consumers threads waiting...")
#Create X threads
for i in range(max_num_threads):
    thread = threading.Thread(target=consumerDetectorThread)
    thread.start()
    threads_array.append(thread)

sk = drawer.getEmptyFrame()
anglesframe = drawer.getEmptyFrame()

windowName = "Detected"
cv2.namedWindow(windowName)
cv2.moveWindow(windowName, 10, 10)
cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

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

            if sk is not None:
                frame = np.concatenate((frame, sk), axis=1)

            if pointsQueue.qsize() > 0:
                posekeypoints = pointsQueue.get()

                anglesframe, bodyAngles = drawer.getBodyAngles(posekeypoints)
                angles = np.array([list(bodyAngles.values())])
                val = int(nnet.predict(angles))

                msg = ATTACK_STATE[val]

                color = [0, 255, 0] if val == 0 else [255, 0, 0]
                cv2.putText(anglesframe, msg, (130, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            frame = np.concatenate((frame, anglesframe), axis=1)

            cv2.imshow(windowName, frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")