import cv2
import numpy as np
from tkinter import filedialog, Tk, simpledialog

pointsStr = ""
initialimagesdir = "/home/gustavokrug/Documents/attack-detection-opencv/"

# Mouse callback function
def draw_circle(event, x, y, flags,param):
    global pointsStr
    if event == cv2.EVENT_LBUTTONDBLCLK:
        root = Tk()
        root.withdraw()
        answer = simpledialog.askstring("Iniciales del cuerpo", "Ingrese inicial del cuerpo para el punto", parent=root)
        if answer is not None:
            cv2.circle(img, (x, y), 6, (255, 0, 0), -1)
            newPoint = " (" + answer + ":x=" + str(x) + ".y=" + str(y) + ")"
            pointsStr = pointsStr + newPoint
            print(newPoint)

root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=initialimagesdir)

img = cv2.imread(file_path)
img = cv2.resize(img, (220, 360))

cv2.namedWindow("Presione ESC para terminar")
cv2.setMouseCallback("Presione ESC para terminar", draw_circle)

while(1):
    cv2.imshow("Presione ESC para terminar", img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()

if pointsStr != "":
    saveToFile = file_path.split("/")[-2] + ".points"
    print(saveToFile)
    f = filedialog.asksaveasfile(mode='w', defaultextension=".txt", initialdir=initialimagesdir, initialfile=saveToFile)
    if f is not None:
        f.write(pointsStr)
        f.close()