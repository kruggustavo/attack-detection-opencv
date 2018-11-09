import cv2
from tkinter import filedialog, Tk, simpledialog

ARMS = 0
LEGS = 1

def getPortionOfBoddy(mode=-1, x=0, y=0, h=0, w=0):
    if mode == ARMS:
        return y,y+int(h/2),x,x+w
    elif mode == LEGS:
        return y+int(h/2),y+h,x,x+w
    else:
        return y,y+h,x,x+w


#myImageUtils = MyImageUtils()
initialimagesdir = "/home/gustavokrug/Documents/attack-detection-opencv/"
IS_VIDEO_FILE = True

root = Tk()
root.withdraw()
media_file = filedialog.askopenfilename(initialdir=initialimagesdir)

if media_file.endswith(".jpg") or media_file.endswith(".jpeg") or media_file.endswith(".png") or media_file.endswith(".bmp"):
    IS_VIDEO_FILE = False

frame_width = 800
frame_height = 600

sample_width = 520
sample_height = 980

cutted_images_folder = filedialog.askdirectory(initialdir=initialimagesdir)

frame_area = frame_width * frame_height     # Area en pixeles del tamaño del frame

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


humans_count = 0
margins = 10
admited_percentage_image_size = 8              # Porcentaje de tamaño de imagenes positivas
process_each_frame = 1                         # Se procesará cada X frames
frame_count = process_each_frame
humans_count = 0

if IS_VIDEO_FILE == True:
    cap = cv2.VideoCapture(media_file)
else:
    frame = cv2.imread(media_file)

while True:
    r = True;
    if IS_VIDEO_FILE == True:
        r, frame = cap.read()

    roi = None

    if r:
        frame_count = frame_count - 1
        if frame_count == 0:
            frame_count = process_each_frame

            frame = cv2.resize(frame, (frame_width, frame_height))  # Downscale to improve frame rate
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            rects1, weights1 = hog.detectMultiScale(frame)

            for (x, y, w, h) in rects1:
                x = x - 10
                w = w + 10
                y = y + 10
                percentage = (w * h * 100) / (frame_area)
                if percentage > admited_percentage_image_size:
                    y, h, x, w = getPortionOfBoddy(-1, x, y, h, w)
                    print(y,h,x,w)
                    roi = frame[y:h, x:w]
                    r = 600.0 / roi.shape[1]
                    dim = (600, int(int(roi.shape[0] * r) / 10) * 10)
                    roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)

                    outputfilename = cutted_images_folder + ("/frame%d.jpg" % humans_count)
                    print(outputfilename)
                    humans_count += 1
                    print("Humanos detectados :" + str(humans_count))
                    cv2.imwrite(outputfilename, roi)


            cv2.imshow("frame", cv2.resize(frame, (320, 240)))

            if roi is not None:
                cv2.imshow("roi", cv2.resize(roi, (320, 240)))

            k = cv2.waitKey(1)
            if k & 0xFF == ord("q"):  # Exit condition
                break

    if IS_VIDEO_FILE == False:
        k = cv2.waitKey(0)
        break