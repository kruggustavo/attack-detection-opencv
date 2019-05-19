from hog.GatherAnnotations import Annotations
from hog.ObjectDetector import ObjectDetector
import os
from concurrent.futures import ThreadPoolExecutor

class WeaponsDetection(object):
    _objectDetectorArray = []
    _labels = []
    _loadPath = ""

    _structure = {}

    def __init__(self, loadPath=None):
        self._loadPath = loadPath
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

        return predictions, labels


    def getRelevantPoints(self, box):
        (x, y, xb, yb) = box[0][0], box[0][1], box[0][2], box[0][3]

        midX = ((xb - x) / 2) + x
        midY = ((yb - y) / 2) + y

        return midX, midY, x, y, xb, yb