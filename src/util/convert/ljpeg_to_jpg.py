#!/usr/bin/env python
import os
import threading
try:
    import Queue
except:
    import queue as Queue
import time
path = "/home/liang/Documents/DDSM"

class Worker(threading.Thread):
    def __init__(self, q, i):
        super(Worker, self).__init__()
        self.q = q
        self.i = i

    def run(self):
        while not self.q.empty():
            file_name = self.q.get()
            try:
                if ".LJPEG" in file_name:
                    out_path = file_name.split('.LJPEG')[0] + ".jpg"
                    if not os.path.isfile(out_path):
                        print("Thread {} converting file {}".format(self.i, file_name))
                        cmd = './ljpeg.py "{0}" "{1}" --visual --scale 1.0 && rm "{0}"'.format(file_name, out_path)
                        # result = subprocess.check_output([sys.executable, './ljpeg.py',file_name, out_path, "--visual", "--scale", "1.0", "&>/dev/null"
                        # cmd = './ljpeg.py "{0}" "{1}" --visual --scale 1.0'.format(file_name, out_path)
                        return_val = os.system(cmd)
                        # if not return_val:
                        #     os.remove(file_name)
                    else:
                        os.remove(file_name)
                elif ".16_pgm" in file_name.lower():
                    os.remove(file_name)
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(e)
                time.sleep(1)
        return

def main():
    q = Queue.Queue()

    try:
        for root, _, file_names in os.walk(path):
            for file_name in file_names:
                if any([string in file_name.lower() for string in [".16_pgm", ".ljpeg"]]):
                    q.put(os.path.join(root, file_name))
        workers = []
        for i in range(5):
            w = Worker(q, i)
            workers.append(w)
            w.start()

        for w in workers:
            w.join()

    except KeyboardInterrupt:
        for w in workers:
            w.join()
if __name__ == '__main__':
    main()
