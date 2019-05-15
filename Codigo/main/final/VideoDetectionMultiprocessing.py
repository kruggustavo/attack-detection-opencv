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
PORTNUM = 55449
AUTHKEY = b'shufflin'

# Estados
ATTACK = 1
NO_ATTACK = 0

# Mensajes en pantalla
ATTACK_STATE = {}
ATTACK_STATE[NO_ATTACK] = "Pose no compatible con ataque"
ATTACK_STATE[ATTACK] = "Pose compatible con ataque!"

workpath = "/home/usuario/Documentos/attack-detection-opencv"
videoFile = workpath +"/Videos/2.mp4"
cap = cv2.VideoCapture(videoFile)
video_width = 420
video_height = 300

# Dibujador y extractor de angulos
drawer = Drawer()

# Red neuronal de angulos
EPOCHS = 40
nnet = NeuralNetwork("trainingangles.csv", 8)
nnet.trainNetwork(EPOCHS)

#framesQueue = queue.Queue()                     # Frames para procesar

netOutput = 0                                   # 1 es agresion
Xseconds = 10                                   # Cantidad de segundos que deben transcurrir para repetir el mensaje de agresion

emptyFrame = np.zeros((video_height, video_width, 3), np.uint8)
skeletonFrame = emptyFrame

mixer.init()
mixer.music.load("47528846.mp3")
agresionTime = time.time() - Xseconds

hasFrame = True
lastVal = NO_ATTACK

def playWarningMsg():
    global lastVal, agresionTime, mixer
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

        # Se procesa una vez que haya respuesta de hilos
        if shared_result_q.qsize() > 0:
            # Enviar frame a cola de trabajo para clientes
            shared_job_q.put(frame)

            # Datos recibidos del cliente
            points = shared_result_q.get()
            print(points)

            skeletonFrame = drawer.drawMultipleSkeletonPoints(emptyFrame.copy(), points)

        frame = np.concatenate((frame, skeletonFrame), axis=1)

        cv2.imshow("Camara", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

print("end")