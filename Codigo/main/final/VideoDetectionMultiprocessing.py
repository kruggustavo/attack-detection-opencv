from final.NeuralNetwork import NeuralNetwork
from final.Drawer import Drawer
from pygame import mixer
from multiprocessing.managers import SyncManager
import queue
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

workpath = "/home/usuario/Documentos/attack-detection-opencv"
img_file = workpath + "/Imagenes/seniors-walking.jpg"
videoFile = workpath +"/Videos/3.mp4"
cap = cv2.VideoCapture(0)
video_width = 360
video_height = 240

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

emptyFrame = np.zeros((video_height, video_width, 3), np.uint8)
skeletonFrame = emptyFrame

mixer.init()
mixer.music.load("47528846.mp3")
agresionTime = time.time() - Xseconds

hasFrame = True
lastVal = NO_ATTACK

def playWarningMsg():
    global lastVal, agresionTime, mixer, netOutput
    # Si han pasado segundos desde el ultimo ataque, habilitar alertas
    if lastVal == ATTACK and (time.time() - agresionTime) > Xseconds:
        lastVal = netOutput

    # Si agresion existe luego de X segundos emitir audio
    if (time.time() - agresionTime) > Xseconds:
        mixer.music.play()
        agresionTime = time.time()
        lastVal = netOutput

    # Alerta visual durante la mitad del tiempo de ataque
    if lastVal == ATTACK and (time.time() - agresionTime) < (Xseconds * 0.5):
        cv2.circle(skeletonFrame, (video_width - 30, video_height - 30), 30, (0, 0, 255), -1)


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

# Enviamos un primer frame para activar el mecanismo de envio: solo se envian datos cuando el cliente haya respondido
hasFrame, frame = cap.read()
frame = cv2.resize(frame, (video_width, video_height))
shared_job_q.put(frame)
print("Sending data to threads. Starting.")
while True:
    hasFrame, frame = cap.read()
    #frame = cv2.imread(img_file)
    if hasFrame:

        frame = cv2.resize(frame, (video_width, video_height))

        # Enviar frame a cola de trabajo para clientes solo si cola esta vacia
        if shared_job_q.qsize() == 0:
            shared_job_q.put(frame)

        # Se procesa una vez que haya respuesta de hilos
        if shared_result_q.qsize() > 0:

            # Datos recibidos del cliente
            pointsFromAllHumans = shared_result_q.get()

            skeletonFrame = emptyFrame.copy()
            netOutput = NO_ATTACK

            # Recorremos grupos de puntos de cada humano
            for pointsSingleHuman in pointsFromAllHumans:
                skeletonFrame = drawer.drawSkeletonPoints(skeletonFrame, pointsSingleHuman)

                angles, lines = drawer.getBodyAngles(pointsSingleHuman)

                if len(angles) > 0:
                    print(angles)
                    angles = np.array([list(angles.values())])
                    try:
                        netOutput = nnet.predict(angles)
                    except:
                        pass

                # Dibujamos tronco
                if "trunkPoints" in lines:
                    pointA = lines["trunkPoints"][0]
                    pointB = lines["trunkPoints"][1]
                    cv2.line(skeletonFrame, pointA, pointB, (100, 7, 65), 3, lineType=cv2.LINE_AA)

            if netOutput == ATTACK:
                playWarningMsg()

            cv2.putText(skeletonFrame, ATTACK_STATE[netOutput],
                        (int((video_width / 2) - (len(ATTACK_STATE[netOutput]) * 5)), 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


        #skeletonFrame = cv2.addWeighted(frame, 0.3, skeletonFrame, 0.7, 0)

        frame = np.concatenate((frame, skeletonFrame), axis=1)

        cv2.imshow("Camara", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")