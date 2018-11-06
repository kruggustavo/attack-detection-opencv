import queue
import threading
import cv2
import time


qBruteImages = queue.Queue()

max_num_threads = 4
threads_array = []


# Specify the paths for the 2 files
protoFile = "/home/gustavokrug/Documents/Documentacion/Models/coco/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "/home/gustavokrug/Documents/Documentacion/Models/coco/pose_iter_160000.caffemodel"

# video size
video_width = 480
video_height = 320
workpath = "/home/gustavokrug/Documents/attack-detection-opencv"
video_file = workpath +"/Videos/3.mp4"
inWidth = 200
inHeight = 200


process_each_frame = 5                         # Se procesará cada X frames
frame_count = process_each_frame
print(video_file)
cap = cv2.VideoCapture(video_file)

def consumerGetNetData():
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    print("Consumer waiting...")
    while True:
        element = qBruteImages.get()
        if element is None:
            return
        inpBlob = cv2.dnn.blobFromImage(element, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        net.forward()
        print("Image processed")
        cv2.imshow('Output-Skeleton', element)
        qBruteImages.task_done()

#Create X threads
for i in range(max_num_threads):
    thread = threading.Thread(target=consumerGetNetData)
    thread.start()
    threads_array.append(thread)

print("Durmiendo por algunos segundos para que los hilos se inicializen...")
time.sleep(5)

# Producer!
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    frame_count = frame_count - 1
    if frame_count == 0:
        frame_count = process_each_frame

        frame = cv2.resize(frame, (video_width, video_height))
        qBruteImages.put(frame)
        print("Image in queue")


qBruteImages.join() #Block until all elements are processed



#Finalize Threads
#Send none to threads to finalize them
for i in range(max_num_threads):
    qBruteImages.put(None)

for thread in threads_array:
    thread.join()