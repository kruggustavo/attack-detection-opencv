from hog.GatherAnnotations import Annotations
from hog.ObjectDetector import ObjectDetector
import os
from concurrent.futures import ThreadPoolExecutor

class BodyPartsDetection(object):
    _objectDetectorArray = []
    _labels = []
    _loadPath = ""

    _structure = {}                 # Estructura: {'arms1', [ {'c', {x, y}}, {'a', {x, y}} ] }

    def __init__(self, loadPath=None, properties=None, keyPoints=None):
        print("Body multiparts detection started")
        print("Parent folder to process: '" + loadPath + "'")
        print("Starting...")
        self._propertiesPath = properties
        self._loadPath = loadPath
        self._keyPoints = keyPoints
        self._loadFolders()

    def _loadFolders(self):
        for folder in os.listdir(self._loadPath):
            print()
            print("Loading resources from '" + self._loadPath + folder + "'")
            self._loadFilesFromFolder(folder)
            print("Ready.")
        print("Sub folders scanning done.")
        print()

    def _loadFilesFromFolder(self, folderLabel):
        fullPath = self._loadPath + folderLabel

        annotationsObj = Annotations(dataset_path=fullPath)
        annotations, imagePaths, label = annotationsObj.annotate()

        detector = ObjectDetector()
        detector.hog_descriptors(imagePaths, annotations, visualizeHog=False)

        propertiesFile = self._propertiesPath + folderLabel + ".properties"

        # Parámetros del detector
        try:
            with open(propertiesFile, "r") as ins:
                for line in ins:
                    key, value = line.split("=")
                    if key == "detection_window_size":
                        v = value.split("*")
                        detector.setDetectWindowSize(int(v[0]) * int(v[1]))
                    if key == "epsilon":
                        detector.setEpsilon(int(value))

            print("Found properties file for '" + folderLabel + "'... properties loaded!")
        except:
            print("Not found properties file for '" + folderLabel + "'")
            pass

        # Carga datos de puntos del cuerpo
        keyPointsFile = self._keyPoints + folderLabel + ".points"

        kpoints = []
        try:
            with open(keyPointsFile, "r") as ins:
                print("Found keypoints file for '" + folderLabel + "'... properties loaded!")
                for line in ins:
                    line = line.replace(" ", "")
                    pairs = line.split(")(")
                    for pair in pairs:
                        pr = pair.split(":")
                        key = pr[0].replace("(", "")        # Point id
                        val = pr[1].replace(")", "")        # Valores

                        # Posicion del keypoint
                        pos = {"x": val.split(".")[0].split("=")[1], "y": val.split(".")[1].split("=")[1]}
                        kp = {key: pos}

                        kpoints.append(kp)

            self._structure[folderLabel] = kpoints
        except:
            print("Not found keypoints file for '" + folderLabel + "'")
            pass

        self._labels.append(folderLabel)
        self._objectDetectorArray.append(detector)

    def detect(self, frame):
        predictions = []
        labels = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            lblkeypoints = []
            for detector, label in zip(self._objectDetectorArray, self._labels):
                future = executor.submit(detector.detect, (frame))
                preds = future.result()
                if len(preds) > 0:
                    predictions.append(preds)
                    labels.append(label)
                    lblkeypoints.append(self._structure[label])

        return predictions, labels, lblkeypoints

    def determineHumanBody(self, boxes, labels):
        fullBodyBoxes = []

        for part1, label1 in zip(boxes, labels):
            if label1.startswith("arms"):
                armX, armY, ax, ay, axb, ayb = self.getRelevantPoints(part1)            # brazos
                for part2, label2 in zip(boxes, labels):
                    if label2.startswith("legs"):
                        legX, legY, lx, ly, lxb, lyb = self.getRelevantPoints(part2)    # piernas
                        if armX >= lx and armX <= lxb:                                  # punto medio de brazos entre margenes de piernas
                            supposedBodyHeight = ((ayb - ay) * 3) + ay                  # altura del cuerpo es supuestamente 3 veces el area de brazos
                            if legY <= supposedBodyHeight:                              # piernas se encuentran dentro del area del cuerpo
                                fullBodyBoxes.append([part1, part2])                    # brazos + piernas = cuerpo completo

        return fullBodyBoxes

    def getRelevantPoints(self, box):
        (x, y, xb, yb) = box[0][0], box[0][1], box[0][2], box[0][3]

        midX = ((xb - x) / 2) + x
        midY = ((yb - y) / 2) + y

        return midX, midY, x, y, xb, yb