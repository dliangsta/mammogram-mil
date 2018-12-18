import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import threading
import shutil
try:
  import Queue
except:
  import queue as Queue
import time
from glob import glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

d = {
  '0': (-1, 0),
  '1': (-1, 1),
  '2': ( 0, 1),
  '3': ( 1, 1),
  '4': ( 1, 0),
  '5': ( 1,-1),
  '6': ( 0,-1),
  '7': (-1, -1)
}

path = "/home/liang/Documents/DDSM/"
new_data_path = "../../data/ddsm/"


class Worker(threading.Thread):

  def __init__(self, q, i):
    super(Worker, self).__init__()
    self.q = q
    self.i = i

  def run(self):
    while not self.q.empty():
      if self.q.empty():
        return
      root, file_name = self.q.get()
      full_file_name = os.path.join(root, file_name)
      delete_folder = root.split('/')[-2]
      try:
        if '.jpg' == os.path.splitext(file_name)[1] or ('.overlay' in file_name.lower() and 'cancer' in root): #and os.path.isfile(full_file_name.replace('.jpg','.OVERLAY')):
          new_file_name = os.path.join(new_data_path, full_file_name.replace(path, '').replace(delete_folder, '')).replace('.jpg','.png')
          if 'cancer_' in new_file_name or 'cancers/cancer_' in new_file_name:
            print('!!', root, file_name, new_file_name)
          
          process = False
          if os.path.isfile(new_file_name):
            if os.path.getsize(new_file_name) == 0:
              process = True
          else:
            process = True
          if process:
            print("Thread {} processing file {} and saving to file {}".format(self.i, full_file_name, new_file_name))
            if 'jpg' in file_name:
              img = imageio.imread(full_file_name, pilmode="RGB")
              # add_mask(img, full_file_name)
              # equalize_hist(img)
              # img = find_rows_and_cols_with_color(img)

              # img = scale_to_shorter_side(img, 300)
              img = resize(img, 299)
              img = np.expand_dims(img[...,0],axis=2)

              if not os.path.exists(os.path.dirname(new_file_name)):
                os.makedirs(os.path.dirname(new_file_name))
              imageio.imwrite(new_file_name, img)
            else:
              # orig_file_name = os.path.join(path, file_name)
              orig_file_name = '/'.join([root,file_name])
              # print(root, file_name, orig_file_name, new_file_name, os.path.exists(orig_file_name))
              print(orig_file_name)
              shutil.copyfile(orig_file_name, new_file_name)
          else:
            if os.path.getsize(new_file_name) == 0:
              print(new_file_name)
      except KeyboardInterrupt:
        return
      except Exception as e:
        print(e)
        # print('bad: {}'.format(full_file_name))
        time.sleep(1)


def preprocess():
  q = Queue.Queue()
  workers = []
  for root, _, file_names in sorted(os.walk(path)):
    for file_name in file_names:
      q.put((root, file_name))
  for i in range(100):
    w = Worker(q, i)
    workers.append(w)
    w.start()
  for w in workers:
    w.join()

  # assert that each cancer case has at least one overlay file
  for case in glob(path + '*'):
    # print(case, glob(case + '/*'))
    # print(case, any('.OVERLAY' in file_name for file_name in  glob(case + '/*')))
    if not any('.OVERLAY' in file_name for file_name in  glob(case + '/*')):
      print(case)


def scale_to_shorter_side(img, shorter_side_length):
  h,w,_ = img.shape
  scale = shorter_side_length / min(h,w)
  new_h,new_w = int(h*scale),int(w*scale)
  img = cv2.resize(img,(new_w,new_h))
  return img

def resize(img, side_length):
  return cv2.resize(img, (side_length, side_length))


def equalize_hist(img):
  for c in range(3):
    img[:,:,c] = cv2.equalizeHist(img[:,:,c])


def find_rows_and_cols_with_color(img):
  old_shape = img.shape
  rows_found = np.array(np.where(np.mean(img, axis=(1,2)) > 20))[0]
  img = img[rows_found,:,:]
  cols_found = np.array(np.where(np.mean(img, axis=(0,2)) > 20))[0]
  img = img[:,cols_found,:]
  new_shape = img.shape
  print(np.array(old_shape)-np.array(new_shape))
  return img


def add_mask(img, file_name):
  with open(file_name.replace('.jpg','.OVERLAY') ,"r") as f:
    lines = f.readlines()
    boundary_seen = False
    for line in lines:
      if not boundary_seen:
        if "BOUNDARY" in line:
          boundary_seen = True
      else:
        # print(line)
        chain = line.strip().split(' ')
        curr = np.array((int(chain[1]),int(chain[0])))
        for direction in chain[2:-1]:
          delta = np.array(d[direction])
          # print(delta, curr)
          curr += np.array(delta)
          img[curr[0]-10:curr[0]+10,curr[1]-10:curr[1]+10,0] = 0
          # print(img[curr[0]])

if __name__ == '__main__':
  preprocess()
    