import numpy as np
import cv2
from imutils.paths import list_images

class Annotations:

    _dataset_path = ""

    def __init__(self, dataset_path=""):
        type(self)._dataset_path = dataset_path

    @classmethod
    def annotate(self):
        annotations = []
        imPaths = []

        #loop through each image and collect annotations
        for imagePath in list_images(self._dataset_path):
            imagePath = imagePath.replace("\ ", " ").replace("\\", "/")
            label = imagePath.split("/")[-2]
            image = cv2.imread(imagePath)
            height, width = image.shape[:2]
            annotations.append([0, 0, int(width), int(height)])
            imPaths.append(imagePath)

        #save annotations and image paths to disk
        annotations = np.array(annotations)
        imPaths = np.array(imPaths, dtype="unicode")

        return annotations, imPaths, label

    @classmethod
    def saveAnnotations(self, annotations_path=None):
        np.save(annotations_path, self._annotations)


