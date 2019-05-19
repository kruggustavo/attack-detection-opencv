from imutils.paths import list_images
from PIL import Image
import cv2
import PIL
import time

targetWidth = 400

pathToImages = "/home/usuario/Documentos/attack-detection-opencv/Imagenes/Dataset/right/"
outputDir = "/home/usuario/Documentos/attack-detection-opencv/Imagenes/Dataset/"


# loop through each image and collect annotations
for imagePath in list_images(pathToImages):
    imagePath = imagePath.replace("\ ", " ").replace("\\", "/")
    label = imagePath.split("/")[-1]
    image = cv2.imread(imagePath)
    height, width = image.shape[:2]
    print(label + " --| " + str(width) + ", " + str(height))

    #if width != targetWidth:
    wpercent = (targetWidth / width)
    targetHeight = int(height * float(wpercent))
    targetHeight = 320
    image = cv2.resize(image, (targetWidth, targetHeight))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    outputPath = outputDir + label.replace("png", "jpg").replace("jpeg", "jpg")
    cv2.imwrite(outputPath, image)

    #cv2.imshow("output", image)
    #time.sleep(50.0 / 1000.0)

print("Done.")