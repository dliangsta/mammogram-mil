import os
import sys
import cv2
import csv
import random
import numpy as np
import tensorflow as tf

def load_data(load_type, image_dir, img_shape, keep_prob=1., grayscale=True, normalize=False):
  assert(load_type in ['test', 'dev', 'train'])

  x = []
  y = []

  csv_path = os.path.join(image_dir, load_type + "_examples.csv")
  with open(csv_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    im_means = []
    for line in reader:
      path, label = line
      # path = '../' + path
      dir_root = os.getcwd().replace('/src','/').replace('/util','/')
      path = dir_root + path
      assert(os.path.exists(path))

      if random.random() < keep_prob:
        img = cv2.imread(path)
        img = cv2.resize(img, img_shape)
        img = img / 256. #.astype(np.float32)
        # if grayscale:
        #   img = np.expand_dims(img[:,:,0], 3)
        # img = np.expand_dims(img[...,0],axipython3 s=2)
        im_means.append(np.mean(img))
        # print(img.shape)

        x.append(img)
        y.append(float(label))

  # Less memory intensive way to subtract mean from images
  x = np.array(x)
  if normalize:
    mean = np.mean(im_means)
    x -= mean
  y = np.array(y).reshape(len(y), 1)

  print("Data for {} set with {} examples from {}. {}% have positive labels and {}% have negative labels.".format(load_type, x.shape[0], csv_path, np.mean(y), 1 - np.mean(y)))
  return x, y
  
