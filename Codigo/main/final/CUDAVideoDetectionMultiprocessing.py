import cv2
import numpy as np
import torch
import math
import time
import queue
from pygame import mixer
from threading import Thread
from final.NeuralNetwork import NeuralNetwork
from final.Drawer import Drawer
from final.models.with_mobilenet import PoseEstimationWithMobileNet
from final.modules.keypoints import extract_keypoints, group_keypoints
from final.modules.load_state import load_state
from final.modules.pose import Pose

# Utilizar solo cpu, utiliza CUDA cuando es false
cpu = False
num_threads = 16

# Inputs de la red neuronal de poses
net_input_height_size = 128

workpath = "/home/usuario/Documents/attack-detection-opencv"
img_file = workpath + "/Imagenes/seniors-walking.jpg"
videoFile = workpath +"/Videos/12.MP4"

video_width = 640
video_height = 480

outputVideoSize = (800, 600)

# Estados
ATTACK = 1
NO_ATTACK = 0

# Mensajes en pantalla
ATTACK_STATE = {}
ATTACK_STATE[NO_ATTACK] = "Pose no compatible con ataque"
ATTACK_STATE[ATTACK] = "Pose compatible con ataque!"

# Dibujador y extractor de angulos
drawer = Drawer()

# Estado
netOutput = NO_ATTACK

Xseconds = 15                                   # Cantidad de segundos que deben transcurrir para repetir el mensaje de agresion
Yseconds =  0.001                               # Cantidad de segundos que deben transcurrir para enviar una imagen a cola de trabajo
Xframes = 3                                     # Cantidad mínima de frames en cola para enviar otro frame. Si cantidad es mayor a este valor, no se envian mas frames a la cola
fontFactor = 46.5                               # Factor de multiplicacion para tamaño de fuente

XsecondsForAgression = 4                        # Si ocurren Y frames de agresiones en X segundos, considerar agresion
YagressionFrames = 5                            # Deben ocurrir Y frames de agresiones en X segundos
blockAgressionCounter = 0
agressionBlockTime = time.time()

# Frames vacios para dibujar esqueletos
emptyFrame = np.zeros((video_height, video_width, 3), np.uint8)
skeletonFrame = emptyFrame.copy()
frameToProcess = emptyFrame.copy()

# Reproductor de audio
mixer.init()
mixer.music.load("47528846.mp3")

# Tiempo transcurrido desde la ultima agresion
agresionTime = time.time() - Xseconds

hasFrame = True
agressionExpired = True

# Margen de area de arma detectada para verificacion
weaponBoxMargin = 50

# Armas
cascadeGunsClassifier = cv2.CascadeClassifier("guns.xml")
weaponInHands = False

# Colas de trabajo
job_q = queue.Queue()
result_q = queue.Queue()

# Objeto dibujante y extractor de angulos y lineas
drawer = Drawer()

# Frame vacio para impresion de mensajes al usuario
board = np.zeros((video_height, video_width * 2, 3), np.uint8)
currentboard = board.copy()
boardFontSize = 0.3
Ytext = (boardFontSize * fontFactor)            # Posicion donde empezar a imprimir texto

# Red neuronal de angulos
mustTrain = False
anglesNet = NeuralNetwork(8)
if mustTrain == True:
    EPOCHS = 5
    anglesNet.loadTrainingSamples("trainingangles_all_cuda_25.csv")
    anglesNet.trainNetwork(EPOCHS)
    anglesNet.saveModel("model_all_25.json")
    quit(0)
else:
    anglesNet.loadModel("model_all_25.json")

# Parametros del detector de poses
pad_value = (0, 0, 0)
img_mean = (128, 128, 128)
img_scale = 1 / 256
stride = 8
upsample_ratio = 4
num_keypoints = Pose.num_kpts
poseNet = PoseEstimationWithMobileNet()
checkpoint = torch.load("checkpoint_iter_370000.pth", map_location='cpu')
load_state(poseNet, checkpoint)
poseNet = poseNet.eval()
if not cpu:
    poseNet = poseNet.cuda()

def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img

def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad

def printText(text, color):
    global Ytext, currentboard
    Ytext = Ytext + (boardFontSize * fontFactor)
    height, width, _ = currentboard.shape
    if Ytext >= height:
        Ytext = (boardFontSize * fontFactor)
        currentboard = board.copy()

    cv2.putText(currentboard, text, (5, int(Ytext)), cv2.FONT_HERSHEY_SIMPLEX, boardFontSize, color, lineType=cv2.LINE_AA)

def attackHandler(frame, lines):
    global agresionTime, weaponInHands, blockAgressionCounter, agressionBlockTime
    agressionConfirmed = False

    blockAgressionCounter = blockAgressionCounter + 1
    # Dentro del tiempo
    if time.time() <= (agressionBlockTime + XsecondsForAgression):
        if YagressionFrames - blockAgressionCounter > 0:
            printText("Quedan " + str(YagressionFrames - blockAgressionCounter) + " para confirmar agresion.", (255,255,255))

        # Tenemos Y agresiones contadas en X segundos: confirmar agresion
        if blockAgressionCounter == YagressionFrames:
            blockAgressionCounter = 0
            agressionConfirmed = True

        # Si falta 1 para confirmar, alargar tiempo
        #if YagressionFrames - blockAgressionCounter == 1:
        #    agressionBlockTime = agressionBlockTime + XsecondsForAgression

    # Fuera del tiempo
    else:
        blockAgressionCounter = 0
        agressionBlockTime = time.time()
        printText("Falso positivo unico.", (255,255,255))


    if agressionConfirmed == True:
        printText(ATTACK_STATE[netOutput], (0, 95, 255))

        arms = ["leftForearmPoints", "rightForearmPoints"]
        if agressionExpired == True:
            agresionTime = time.time()

        # Verifica armas en frame cercanos a manos
        results = cascadeGunsClassifier.detectMultiScale(cv2.resize(frame, outputVideoSize), 1.5, 5, minSize=(100, 100))
        if len(results) > 0:
            weaponInHands = False
            # Agregamos margenes a imagenes de armas
            for (x, y, w, h) in results:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
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

        if weaponInHands == True:
            printText("Arma de fuego detectada con pose compatible. Posible agresion en curso.", (66, 66, 255))
            mixer.music.play()


def worker(job_q, result_q):
    global poseNet, upsample_ratio, num_keypoints, stride, img_scale, img_mean, pad_value
    while True:
        # Obtenemos frame del servidor de hilos
        img = job_q.get()
        angles = {}

        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu:
            tensor_img = tensor_img.cuda()

        stages_output = poseNet(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        # n = cantidad de points sets (personas distintas)
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])

            labeledKeypoints = drawer.getLabeledPoints(pose.getOrderedKeypoints())

            my_list = []
            my_list.append(labeledKeypoints)
            my_list.append(img)
            result_q.put(my_list)


for i in range(num_threads):
    t = Thread(target=worker, args=(job_q, result_q))
    t.setDaemon(True)
    t.start()

cap = cv2.VideoCapture(videoFile)   # videoFile

empty = np.zeros((video_height, video_width, 3), np.uint8)
out = empty.copy()

pointTime = time.time() + Yseconds
agressionBlockTime = time.time()

windowName = "Camara"
cv2.namedWindow(windowName);
cv2.moveWindow(windowName, 20,20);
printText("Inicio...", (255, 255, 255))
was_read, img = cap.read()
while True:
    netOutput = NO_ATTACK
    weaponInHands = False

    try:
        was_read, img = cap.read()
        img = cv2.resize(img, (video_width, video_height))
    except:
        pass

    agressionExpired = ((time.time() - agresionTime) > Xseconds)

    if (time.time() - pointTime >= Yseconds and job_q.qsize() <= Xframes) or result_q.qsize() > 0:
        pointTime = time.time()
        job_q.put(img)

    if result_q.qsize() > 0:
        my_list = result_q.get()
        labeledKeypoints = my_list[0]
        processedFrame = my_list[1]

        angles, lines = drawer.getBodyAngles(labeledKeypoints)
        out = drawer.drawSkeletonLines(processedFrame, lines)

        if len(angles) > 0:
            angles = np.array([list(angles.values())])
            printText(str(angles), (0, 255, 0))
            try:
                netOutput = int(anglesNet.predict(angles))
            except Exception as e:
                print(str(e))
                pass

            if netOutput == ATTACK:
                attackHandler(out, lines)

    try:
        img = np.concatenate((img, out), axis=1)
        img = np.concatenate((img, currentboard), axis=0)
    except:
        pass

    cv2.imshow(windowName, cv2.resize(img, ((int) (video_width * 2), (int) (video_height * 2) )))

    key = cv2.waitKey(33)
    if key == 27:  # esc
        break