import cv2
import numpy as np
from tkinter import filedialog, Tk, simpledialog


# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:

        root = Tk()
        root.withdraw()
        answer = simpledialog.askstring("Iniciales del cuerpo", "Ingrese inicial del cuerpo para el punto", parent=root)

        if answer is not None:
            cv2.circle(img, (x, y), 6, (255, 0, 0), -1)
            print(answer + " " + str(x) + ", " + str(y))

root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

img = cv2.imread(file_path)
img = cv2.resize(img, (220, 360))

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
