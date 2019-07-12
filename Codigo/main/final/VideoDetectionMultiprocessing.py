from final.NeuralNetwork import NeuralNetwork
from final.Drawer import Drawer
from pygame import mixer
from multiprocessing.managers import SyncManager
import queue
import cv2
import numpy as np
import time
import random
import threading

# Attack detection
# Detector de ataques y amenazas humanas en imagenes secuenciales mediante el entrenamiento de redes neuronales
# Trabajo final de Grado para la obtencion del titulo de Ingeniero Informatico
# Autor: Gustavo Krug
# Universidad Catolica Nuestra Señora de la Asuncion, 2019
#

print("Human attack detection")

# Parametros generales
# Servidor de hilos multiprocesamiento
IP = 'localhost'
PORTNUM = 55141
AUTHKEY = b'shufflin'

# Estados
ATTACK = 1
NO_ATTACK = 0

# Mensajes en pantalla
ATTACK_STATE = {}
ATTACK_STATE[NO_ATTACK] = "Pose no compatible con ataque"
ATTACK_STATE[ATTACK] = "Pose compatible con ataque!"

#workpath = "/home/usuario/Documentos/attack-detection-opencv"
#img_file = workpath + "/Imagenes/seniors-walking.jpg"
#videoFile = workpath +"/Videos/5.mp4"
cap = cv2.VideoCapture(0)
video_width = 320
video_height = 200

# Dibujador y extractor de angulos
drawer = Drawer()

# Red neuronal de angulos
EPOCHS = 1
nnet = NeuralNetwork(8)
#nnet.loadTrainingSamples("trainingangles.csv")
#nnet.trainNetwork(EPOCHS)
nnet.loadModel("model.json")
#nnet.saveModel("model.json")

netOutput = NO_ATTACK
Xseconds = 10                                   # Cantidad de segundos que deben transcurrir para repetir el mensaje de agresion
Yseconds = 0.5                                  # Cantidad de segundos que deben transcurrir para enviar una imagen a cola de trabajo

Xframes = 5                                     # Cantidad mínima de frames en cola para enviar otro frame. Si cantidad es mayor a este valor, no se envian mas frames a la cola

emptyFrame = np.zeros((video_height, video_width, 3), np.uint8)
skeletonFrame = emptyFrame
frameToProcess = emptyFrame

mixer.init()
mixer.music.load("47528846.mp3")
agresionTime = time.time() - Xseconds

hasFrame = True
agressionExpired = True

# Margen de area de arma detectada para verificacion
weaponBoxMargin = 50

# Armas
cascadeGunsClassifier = cv2.CascadeClassifier("guns.xml")
weaponInHands = False

def attackHandler(frame, lines):
    global agresionTime, weaponInHands
    arms = ["leftForearmPoints", "rightForearmPoints"]
    if agressionExpired == True:
        agresionTime = time.time()
        mixer.music.play()

    # Verifica armas en frame cercanos a manos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    results = cascadeGunsClassifier.detectMultiScale(frame, 1.5, 5, minSize=(100, 100))
    if len(results) > 0:
        weaponInHands = False
        # Agregamos margenes a imagenes de armas
        for (x, y, w, h) in results:
            x -= weaponBoxMargin
            y -= weaponBoxMargin
            w += weaponBoxMargin
            h += weaponBoxMargin
            for arm in arms:
                if arm in lines:
                    hx, hy = lines[arm][1] # Puntos de la mano
                    # Si coordenadas de manos estan dentro del area de arma detectada
                    if hx >= x and hx <= (x + w) and hy >= y and hy <= (y + h):
                        weaponInHands = True

# Servidor de hijos para multiprocesamiento
job_q = queue.Queue()
result_q = queue.Queue()

class JobQueueManager(SyncManager):
    pass

JobQueueManager.register('get_job_q', callable=lambda: job_q)
JobQueueManager.register('get_result_q', callable=lambda: result_q)

manager = JobQueueManager(address=('', PORTNUM), authkey=AUTHKEY)
manager.start()

shared_job_q = manager.get_job_q()
shared_result_q = manager.get_result_q()

# Enviamos un primer frame para activar el mecanismo de envio: solo se envian datos cuando la cola esta vacia
hasFrame, frame = cap.read()
frame = cv2.resize(frame, (video_width, video_height))
shared_job_q.put(frame)
print("Sending data to threads. Starting.")

# Hilo para deteccion de armas en escena
#thread = threading.Thread(target=weaponsDetectionConsumer)
#thread.start()

frameId = 0

# Diccionario que contiene frames que estan siendo procesados identificados con un numero de id
localFramesDict = {}
pointTime = time.time()
while True:
    #time.sleep(100 / 1000)
    frameId = random.randint(0, 999999)
    hasFrame, frame = cap.read()
    #frame = cv2.imread(img_file)
    if hasFrame:

        frame = cv2.resize(frame, (video_width, video_height))

        if (time.time() - agresionTime) > Xseconds:
            agressionExpired = True
        else:
            agressionExpired = False


        # Enviar frame e ID a cola de trabajo para clientes cada Y segundos solo si cola tiene no mas de X frames esperando
        if (time.time() - pointTime) >= Yseconds:
            pointTime = time.time()
            if shared_job_q.qsize() <= Xframes:
                packetData = {}
                packetData[frameId] = frame
                shared_job_q.put(packetData)

                localFramesDict[frameId] = frame

        # Se procesa una vez que haya respuesta de hilos (se espera un dict con el id del frame procesado y lista de puntos)
        if shared_result_q.qsize() > 0:
            packetData = shared_result_q.get()
            returnedFrameId = list(packetData)[0]

            # Datos recibidos del cliente
            pointsFromAllHumans = packetData[returnedFrameId]

            netOutput = NO_ATTACK

            # Recorremos grupos de puntos de cada humano
            for pointsSingleHuman in pointsFromAllHumans:
                skeletonFrame = emptyFrame.copy()

                angles, lines = drawer.getBodyAngles(pointsSingleHuman)
                skeletonFrame = drawer.drawSkeletonLines(skeletonFrame, lines)
                if len(angles) > 0:
                    print(angles)
                    angles = np.array([list(angles.values())])
                    try:
                        netOutput = int(nnet.predict(angles))
                    except:
                        pass
                    if netOutput == ATTACK:
                        print("Attack!")
                        frameToProcess = localFramesDict[returnedFrameId]
                        attackHandler(frameToProcess, lines)

            cv2.putText(skeletonFrame, ATTACK_STATE[netOutput], (int((video_width / 2) - (len(ATTACK_STATE[netOutput]) * 5)), 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            if weaponInHands == True and netOutput == ATTACK:
                cv2.circle(skeletonFrame, (video_width - 30, video_height - 30), 25, (0, 0, 255), -1)

            skeletonFrame = cv2.addWeighted(frame, 0.3, skeletonFrame, 0.7, 0)

        frame = np.concatenate((frame, skeletonFrame), axis=1)

        cv2.imshow("Camara", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")
