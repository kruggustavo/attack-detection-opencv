# Eli Bendersky [http://eli.thegreenplace.net]
# This code is in the public domain.
from __future__ import print_function
from multiprocessing.managers import SyncManager
import multiprocessing
import cv2
import numpy as np

IP = 'localhost'
PORTNUM = 55141
AUTHKEY = b'shufflin'

frameWidth = frameHeight = 0

weightsFile = "pose/mpi/pose_iter_150000.caffemodel"
#protoFile = "pose/coco/pose_deploy_linevec.prototxt"
protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"

nPoints = 18

# Referencias
# c = cabeza, n = cuello, h = hombro (i, d), c = codo (i, d), m = mano (i, d), p = pecho
# hp = cadera (i, d), r = rodilla(i, d), p = pie (i, d)
POINTS_LABELS = ["c", "n", "hi", "ci", "mi", "hd", "cd", "md", "hpi", "ri", "pi", "hpd", "rd", "pd", "p"]

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

# Segun indices obtenidos de la red neuronal, estas son partes del cuerpo correspondientes
humanPartsDict = {
    3:"leftForearmPoints",
    5:"rightForearmPoints",
    2:"leftArmPoints",
    4:"rightArmPoints",
    7:"leftThighPoints",
    10:"rightThighPoints",
    8:"leftLegPoints",
    11:"rightLegPoints",
    12:"neck"
}
# Puntos que conforman las partes del cuerpo
humanPointsFromPartsDict = {
    "leftForearmPoints": ("ci", "mi"),
    "rightForearmPoints": ("cd", "md"),
    "leftArmPoints": ("hi", "ci"),
    "rightArmPoints": ("hd", "cd"),
    "leftThighPoints": ("hpi", "ri"),
    "rightThighPoints": ("hpd", "rd"),
    "leftLegPoints": ("ri", "pi"),
    "rightLegPoints": ("rd", "pd"),
    "neck": ("c", "p")
}

detected_keypoints = []
keypoints_list = np.zeros((0,3))

def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, b = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            #print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

def make_client_manager(ip, port, authkey):
    class ServerQueueManager(SyncManager):
        pass

    ServerQueueManager.register('get_job_q')
    ServerQueueManager.register('get_result_q')

    manager = ServerQueueManager(address=(ip, port), authkey=authkey)

    connected = False
    while connected == False:
        try:
            manager.connect()
            connected = True
        except:
            pass


    print('Client connected to %s:%s' % (ip, port))
    return manager


def worker(job_q, result_q):
    global frameHeight, frameWidth, detected_keypoints, keypoints_list, net

    while True:
        try:
            # Obtenemos frame del servidor de hilos
            packetData = job_q.get() #get_nowait()

            frameId = list(packetData)[0]
            frame = packetData[frameId]

            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            detected_keypoints = []
            keypoints_list = np.zeros((0, 3))
            # Fix the input Height and get the width according to the Aspect Ratio
            inHeight = 180
            inWidth = 300

            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)
            output = net.forward()

            detected_keypoints = []
            keypoints_list = np.zeros((0, 3))
            keypoint_id = 0
            threshold = 0.1

            for part in range(nPoints):
                probMap = output[0, part, :, :]
                probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
                keypoints = getKeypoints(probMap, threshold)
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)

            valid_pairs, invalid_pairs = getValidPairs(output)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

            # Cantidad de personas: colocamos un array de puntos por cada persona
            humansPointsLists = list({} for i in range(len(personwiseKeypoints)))

            peopleCount = len(personwiseKeypoints)

            for i in range(17):
                for n in range(peopleCount):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    X = np.int32(keypoints_list[index.astype(int), 0])
                    Y = np.int32(keypoints_list[index.astype(int), 1])
                    point1 = (X[0], Y[0])
                    point2 = (X[1], Y[1])
                    cv2.line(frame, point1, point2, colors[i], 3, cv2.LINE_AA)
                    # Determinamos puntos de la recta
                    if i in humanPartsDict:
                        bodyPartLabel = humanPartsDict[i]
                        labels = humanPointsFromPartsDict[bodyPartLabel]

                        humansPointsLists[n][labels[0]] = point1
                        humansPointsLists[n][labels[1]] = point2

            # Colocamos resultado en cola de resultados para servidor
            result = {}
            result[frameId] = humansPointsLists

            result_q.put(result)
            print("Processed! " + str(frameId) + " " + str(humansPointsLists))
        except Exception as e:
            print(str(e))
            frame = None


print("Client program started")
number_of_threads = 19
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

manager = make_client_manager(IP, PORTNUM, AUTHKEY)
job_q = manager.get_job_q()
result_q = manager.get_result_q()

procs = []
for i in range(number_of_threads):
    p = multiprocessing.Process(target=worker, args=(job_q, result_q))
    procs.append(p)
    p.start()

for p in procs:
    p.join()