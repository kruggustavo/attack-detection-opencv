# Eli Bendersky [http://eli.thegreenplace.net]
# This code is in the public domain.
from __future__ import print_function
from multiprocessing.managers import SyncManager
from final.OpenPoseMutiple import OpenPoseMultiple
import multiprocessing
import queue

IP = 'localhost'
PORTNUM = 55441
AUTHKEY = b'shufflin'

# Red neuronal de poses
op = OpenPoseMultiple(protoFile="pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt", weightsFile="pose/mpi/pose_iter_150000.caffemodel")

def make_client_manager(ip, port, authkey):
    class ServerQueueManager(SyncManager):
        pass

    ServerQueueManager.register('get_job_q')
    ServerQueueManager.register('get_result_q')

    manager = ServerQueueManager(address=(ip, port), authkey=authkey)

    connected = False
    while connected == False:
        try:
            manager.connect()
            connected = True
        except:
            pass


    print('Client connected to %s:%s' % (ip, port))
    return manager


def worker(job_q, result_q):
    while True:
        #try:
        # Obtenemos frame del servidor de hilos
        frame = job_q.get() #get_nowait()
        points = op.detectHumanPose(frame)

        # Colocamos resultado en cola de resultados para servidor
        result_q.put(points)
        print("Processed! " + str(points))
        #except queue.Empty:
        #    return

print("Client program started")
number_of_threads = 4

manager = make_client_manager(IP, PORTNUM, AUTHKEY)
job_q = manager.get_job_q()
result_q = manager.get_result_q()

procs = []
for i in range(number_of_threads):
    p = multiprocessing.Process(target=worker, args=(job_q, result_q))
    procs.append(p)
    p.start()

for p in procs:
    p.join()