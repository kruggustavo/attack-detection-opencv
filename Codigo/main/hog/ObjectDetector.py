import dlib
import cv2

#
# Referencia: http://www.hackevolve.com/create-your-own-object-detector/
#
# Detector de objetos
#

class ObjectDetector(object):
    def __init__(self, options=None, loadPath=None):
        self.options = options
        if self.options is None:
            self.options = dlib.simple_object_detector_training_options()
            self.options.C = 5
            self.options.num_threads = 4
            self.options.detection_window_size = 50 * 45
            self.options.epsilon = 20.00

        if loadPath is not None:
            self._detector = dlib.simple_object_detector(loadPath)
    
    def setEpsilon(self, value):
        self.options.epsilon = value
    
    def setDetectWindowSize(self, value):
        self.options.detection_window_size = value

    def _prepare_annotations(self, annotations):
        annots = []
        for (x, y, xb, yb) in annotations:
            annots.append([dlib.rectangle(left=int(x), top=int(y), right=int(xb), bottom=int(yb))])
        return annots

    def _prepare_images(self, imagePaths):
        images = []
        for imPath in imagePaths:
            image = cv2.imread(imPath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def hog_descriptors(self, imagePaths, annotations, visualizeHog=False, savePath=None):
        annotations = self._prepare_annotations(annotations)
        images = self._prepare_images(imagePaths)
        self._detector = dlib.train_simple_object_detector(images, annotations, self.options)

        if visualizeHog:
            win = dlib.image_window()
            win.set_image(self._detector)
            dlib.hit_enter_to_continue()

        if savePath is not None:
            self._detector.save(savePath)

        return self

    def predict(self, image):
        boxes = self._detector(image, 0)
        return boxes

    def detect(self, image):
        gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        boxes = self.predict(gray_frame)

        preds = []
        for box in boxes:
            (x, y, xb, yb) = [box.left(), box.top(), box.right(), box.bottom()]
            if x > 0 or y > 0:
                preds.append((x, y, xb, yb))

        return preds