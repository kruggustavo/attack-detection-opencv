import argparse

import cv2
import numpy as np
import torch
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

# Red neuronal de angulos
EPOCHS = 1
nnet = NeuralNetwork(8)
#nnet.loadTrainingSamples("trainingangles.csv")
#nnet.trainNetwork(EPOCHS)
nnet.loadModel("model.json")

# Estados
ATTACK = 1
NO_ATTACK = 0

# Mensajes en pantalla
ATTACK_STATE = {}
ATTACK_STATE[NO_ATTACK] = "Pose no compatible con ataque"
ATTACK_STATE[ATTACK] = "Pose compatible con ataque!"

video_width = 320
video_height = 200

# Dibujador y extractor de angulos
drawer = Drawer()

# Red neuronal de angulos
EPOCHS = 1
anglesNet = NeuralNetwork(8)
#anglesNet.loadTrainingSamples("trainingangles.csv")
#anglesNet.trainNetwork(EPOCHS)
anglesNet.loadModel("model.json")
#anglesNet.saveModel("model.json")

netOutput = NO_ATTACK
Xseconds = 10                                   # Cantidad de segundos que deben transcurrir para repetir el mensaje de agresion
Yseconds = 0.6                                    # Cantidad de segundos que deben transcurrir para enviar una imagen a cola de trabajo

Xframes = 3                                     # Cantidad mínima de frames en cola para enviar otro frame. Si cantidad es mayor a este valor, no se envian mas frames a la cola

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

job_q = queue.Queue()
result_q = queue.Queue()

drawer = Drawer()

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

def worker(i, job_q, result_q):
    global poseNet, drawer, anglesNet, NO_ATTACK, ATTACK
    print(str(i) + " thread started..")
    while True:
        netOutput = NO_ATTACK

        # Obtenemos frame del servidor de hilos
        img = job_q.get() #get_nowait()
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

        labeledKeypoints = {}

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

            img  = drawer.drawSkeletonPoints(img, labeledKeypoints)
            angles, lines = drawer.getBodyAngles(labeledKeypoints)

            if len(angles) > 0:
                angles = np.array([list(angles.values())])
                print(angles)
                try:
                    netOutput = int(anglesNet.predict(angles))
                except Exception as e:
                    print(str(e))
                    pass

                if netOutput == ATTACK:
                    print("Attack!")


            # Dibujamos tronco
            if "trunkPoints" in lines:
                pointA = lines["trunkPoints"][0]
                pointB = lines["trunkPoints"][1]
                cv2.line(img, pointA, pointB, (100, 7, 65), 3, lineType=cv2.LINE_AA)




        result_q.put(img)




net_input_height_size = 64
cpu = True
track_ids = True
pad_value = (0, 0, 0)
img_mean = (128, 128, 128)
img_scale = 1 / 256
stride = 8
upsample_ratio = 4
num_keypoints = Pose.num_kpts
previous_poses = []

checkpoint_path = "checkpoint_iter_370000.pth"
poseNet = PoseEstimationWithMobileNet()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
load_state(poseNet, checkpoint)
poseNet = poseNet.eval()
if not cpu:
    net = poseNet.cuda()

num_threads = 18

for i in range(num_threads):
    t = Thread(target=worker, args=(i, job_q, result_q))
    t.setDaemon(True)
    t.start()

minimumStackSize = 5
cap = cv2.VideoCapture(0)
was_read, img = cap.read()
orig_img = img.copy()
img = cv2.resize(img, (video_width, video_height))
out = np.zeros((video_height, video_width, 3), np.uint8)

job_q.put(img)
pointTime = time.time()
while True:
    was_read, img = cap.read()
    img = cv2.resize(img, (video_width, video_height))

    if time.time() - pointTime >= Yseconds:
        job_q.put(img)
        pointTime = time.time()

    if result_q.qsize() > 0 :
        out = result_q.get()

    img = np.concatenate((img, out), axis=1)
    cv2.imshow('Camara', img)


    key = cv2.waitKey(33)
    if key == 27:  # esc
        break