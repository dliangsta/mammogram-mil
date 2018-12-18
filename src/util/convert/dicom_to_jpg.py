import pydicom as dicom
import os
import cv2
import threading
import Queue


path = "/home/liang/Documents/ISPY1"

class Worker(threading.Thread):
    def __init__(self, q, i):
        super(Worker, self).__init__()
        self.q = q
        self.i = i

    def run(self):
        while not self.q.empty():
            try:
                file_name = self.q.get()
                ds = dicom.dcmread(file_name)
                pixel_array_numpy = ds.pixel_array
                out_filename = file_name.replace('.dcm', '.jpg')
                cv2.imwrite(out_filename, pixel_array_numpy)
                os.remove(file_name)
                print('converted {}'.format(file_name))
            except Exception as e:
                print('failed to convert {} {}'.format(file_name, e))
                return

def queue_files(q):
    count = 0
    for root, subFolders, file_names in os.walk(path):
        for file_name in file_names:
            if '.dcm' in file_name.lower():
                count += 1
                q.put(os.path.join(root, file_name))
                ds = dicom.dcmread(os.path.join(root, file_name))
                # return count
    return count

def spin_up_workers(workers, q, n=10):
    for i in range(n):
        w = Worker(q, i)
        workers.append(w)
        w.start()

def main():
    workers = []
    q = Queue.Queue()

    try:
        while True:
            n_queued = queue_files(q)
            spin_up_workers(workers, q, 10)
            if n_queued <= 1:
                return
    except KeyboardInterrupt:
        for w in workers:
            w.join()

if __name__ == '__main__':
    main()