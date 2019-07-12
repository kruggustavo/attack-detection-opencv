import cv2
import numpy as np
import torch
import random
import math
import time
import queue
from threading import Thread
from final.NeuralNetwork import NeuralNetwork
from final.Drawer import Drawer
from pygame import mixer
from final.models.with_mobilenet import PoseEstimationWithMobileNet
from final.modules.keypoints import extract_keypoints, group_keypoints
from final.modules.load_state import load_state
from final.modules.pose import Pose

# Utilizar solo cpu, utiliza CUDA cuando es false
cpu = True

# Inputs de la red neuronal de poses
net_input_height_size = 96

workpath = "/home/usuario/Documentos/attack-detection-opencv"
img_file = workpath + "/Imagenes/seniors-walking.jpg"
videoFile = workpath +"/Videos/12.MP4"

video_width = 360
video_height = 240

outputVideoSize = (880, 500)

combinations = 5000  # Combinaciones distintas generadas

# Estados
ATTACK = 1
NO_ATTACK = 0

POSE = NO_ATTACK

# Mensajes en pantalla
ATTACK_STATE = {}
ATTACK_STATE[NO_ATTACK] = "Pose no compatible con ataque"
ATTACK_STATE[ATTACK] = "Pose compatible con ataque!"

# Dibujador y extractor de angulos
drawer = Drawer()

# Estado
netOutput = NO_ATTACK

Xseconds = 15  # Cantidad de segundos que deben transcurrir para repetir el mensaje de agresion
Yseconds = 2  # Cantidad de segundos que deben transcurrir para enviar una imagen a cola de trabajo
Xframes = 3  # Cantidad mínima de frames en cola para enviar otro frame. Si cantidad es mayor a este valor, no se envian mas frames a la cola
fontFactor = 46.5  # Factor de multiplicacion para tamaño de fuente

# Frames vacios para dibujar esqueletos
emptyFrame = np.zeros((video_height, video_width, 3), np.uint8)
skeletonFrame = emptyFrame.copy()
frameToProcess = emptyFrame.copy()

hasFrame = True

# Margen de area de arma detectada para verificacion
weaponBoxMargin = 50

# Colas de trabajo
job_q = queue.Queue()
result_q = queue.Queue()

# Objeto dibujante y extractor de angulos y lineas
drawer = Drawer()

# Frame vacio para impresion de mensajes al usuario
board = np.zeros((video_height, video_width * 2, 3), np.uint8)
currentboard = board.copy()
boardFontSize = 0.3
Ytext = (boardFontSize * fontFactor)  # Posicion donde empezar a imprimir texto

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

    cv2.putText(currentboard, text, (5, int(Ytext)), cv2.FONT_HERSHEY_SIMPLEX, boardFontSize, color,
                lineType=cv2.LINE_AA)

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
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

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


for i in range(12):
    t = Thread(target=worker, args=(job_q, result_q))
    t.setDaemon(True)
    t.start()

cap = cv2.VideoCapture(videoFile)

empty = np.zeros((video_height, video_width, 3), np.uint8)
out = empty.copy()

pointTime = time.time() + Yseconds

windowName = "Camara"
cv2.namedWindow(windowName);
cv2.moveWindow(windowName, 20, 20);
printText("Inicio...", (255, 255, 255))
printText("Creating " + str(combinations) + " combinations",  (99, 65, 99))
printText("Pose mode " + ("NO" if POSE == NO_ATTACK else "") + " ATACK",  (255, 255, 255))

file = open("trainingangles_output_cuda.csv","w+")
while True:
    netOutput = NO_ATTACK
    weaponInHands = False

    was_read, img = cap.read()

    img = cv2.resize(img, (video_width, video_height))

    if time.time() - pointTime >= Yseconds or result_q.qsize() > 0:
        pointTime = time.time()
        job_q.put(img)

    if result_q.qsize() > 0:
        my_list = result_q.get()
        labeledKeypoints = my_list[0]
        processedFrame = my_list[1]

        originalAngles, lines = drawer.getBodyAngles(labeledKeypoints)
        out = drawer.drawSkeletonLines(processedFrame, lines)

        angles = originalAngles
        angles["pose"] = POSE

        angles = np.array([list(angles.values())])
        anglesStr = str(angles)
        if len(angles) > 0 and "0 0 0 0 0 0 0 0" not in anglesStr:
            anglesStr = anglesStr.replace(". ]]", "]]")
            anglesStr = anglesStr.replace(". ", ".0")
            anglesStr = anglesStr.replace(" ]]", "]]").replace("[[ ", "[[")
            anglesStr = anglesStr.replace(" ]]", "]]").replace("[[ ", "[[")
            anglesStr = anglesStr.replace("  ", " ").replace("  ", " ")
            anglesStr = anglesStr.replace(" ", ",")
            anglesStr = anglesStr.replace("[", "").replace("]", "")

            if len(anglesStr) > 1:
                printText(anglesStr, (0, 255, 0))
                file.write(anglesStr + "\n")

            # Generamos X combinaciones aleatorias parecidas a la lista de angulos anteriores (se suman o restan decimales a cada valor al azar)
            max_variation = 5  # Mayor y Menor variaciones entre angulos
            min_variation = 0.2

            for x in range(combinations):
                newCombinationAngles = []
                for a in range(len(originalAngles) - 1):  # 8 Angulos
                    angle = tuple(originalAngles.items())[a][1]
                    newValue = round(random.uniform(0, min_variation) if angle < max_variation else angle + random.uniform(max_variation * -1, max_variation), 1)
                    newCombinationAngles.append(newValue)

                output = str(newCombinationAngles).replace("]", "").replace("[", "").replace(" ", "")
                if output != "0.0" and output != "":
                    output += "," + str(POSE)
                    file.write(output + "\n")




    img = np.concatenate((img, out), axis=1)
    img = np.concatenate((img, currentboard), axis=0)
    cv2.imshow(windowName, cv2.resize(img, outputVideoSize))

    key = cv2.waitKey(33)
    if key == 27:  # esc
        break