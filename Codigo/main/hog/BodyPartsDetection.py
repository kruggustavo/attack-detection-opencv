from hog.GatherAnnotations import Annotations
from hog.ObjectDetector import ObjectDetector
import os
from concurrent.futures import ThreadPoolExecutor

class BodyPartsDetection(object):
    _objectDetectorArray = []
    _labels = []
    _loadPath = ""

    def __init__(self, loadPath=None, properties=None):
        print("Body multiparts detection started")
        print("Parent folder to process: '" + loadPath + "'")
        print("Starting...")
        self._propertiesPath = properties
        self._loadPath = loadPath
        self._loadFolders()

    def _loadFolders(self):
        for folder in os.listdir(self._loadPath):
            print("Loading resources from '" + self._loadPath + folder + "'")
            self._loadFilesFromFolder(folder)
            print("Ready.")
        print("Sub folders scanning done.")

    def _loadFilesFromFolder(self, folderName):
        fullPath = self._loadPath + folderName

        annotationsObj = Annotations(dataset_path=fullPath)
        annotations, imagePaths, label = annotationsObj.annotate()

        detector = ObjectDetector()
        detector.hog_descriptors(imagePaths, annotations, visualizeHog=False)

        propertiesFile = self._propertiesPath + folderName + ".properties"

        # Carga datos personalizados del detector
        try:
            with open(propertiesFile, "r") as ins:
                for line in ins:
                    key, value = line.split("=")
                    if key == "detection_window_size":
                        v = value.split("*")
                        detector.setDetectWindowSize(int(v[0]) * int(v[1]))
                    if key == "epsilon":
                        detector.setEpsilon(int(value))
            print("Found properties file for '" + folderName + "'... properties loaded!")
        except:
            print("Not found properties file for '" + folderName + "'")
            pass


        self._labels.append(folderName)
        self._objectDetectorArray.append(detector)

    def detect(self, frame):
        predictions = []
        labels = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            for detector, label in zip(self._objectDetectorArray, self._labels):
                future = executor.submit(detector.detect, (frame))
                preds = future.result()
                if len(preds) > 0:
                    predictions.append(preds)
                    labels.append(label)

        return predictions, labels

    def determineHumanBody(self, boxes, labels):
        fullBodyBoxes = []

        for part1, label1 in zip(boxes, labels):
            if label1.startswith("arms"):
                armX, armY, ax, ay, axb, ayb = self.getRelevantPoints(part1)            # arms points
                for part2, label2 in zip(boxes, labels):
                    if label2.startswith("legs"):
                        legX, legY, lx, ly, lxb, lyb = self.getRelevantPoints(part2)    # legs points
                        if armX >= lx and armX <= lxb:                                    # arms X mid point between X margins from legs
                            supposedBodyHeight = ((ayb - ay) * 3) + ay                  # supposed height of body is a space 3 times under arms
                            if legY <= supposedBodyHeight:                               # legs are in supposed height of the body
                                fullBodyBoxes.append([part1, part2])                    # arms and legs = full body

        return fullBodyBoxes

    def getRelevantPoints(self, box):
        (x, y, xb, yb) = box[0][0], box[0][1], box[0][2], box[0][3]

        midX = ((xb - x) / 2) + x
        midY = ((yb - y) / 2) + y

        return midX, midY, x, y, xb, yb