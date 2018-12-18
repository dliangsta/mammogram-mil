import os
import sys
import cv2
import csv
import random
import numpy as np
import tensorflow as tf

from glob import glob

FLAGS = None

class DataIterator:
  
  def __init__(self, config):
    self.config = config
    metadata_filename = self.get_cwd() + 'data/metadata.txt'
    assert os.path.exists(metadata_filename)
    with open(metadata_filename, 'r') as f:
      mean, std = f.readlines()[0].split(',')
      self.mean, self.std = float(mean), float(std)

    self.csv_path = config.image_dir

  def get_ops(self, load_type):
    assert(load_type in ['test', 'dev', 'train'])
    
    # num_epochs = 1 if load_type == 'train' else 1 
    num_epochs = 1

    csv_path = os.path.join(self.csv_path, load_type + "_examples.csv")
    with open(csv_path, 'r') as f:
      num_examples = len(f.readlines()) - 1
      num_minibatches = (num_examples // self.config.batch_size + int(num_examples % self.config.batch_size > 0)) * num_epochs

    dataset = tf.data.experimental.make_csv_dataset(
      csv_path,
      batch_size=1,
      column_names=['Path', 'Label'],
      label_name='Label',
      header=True,
      shuffle=True,
      num_epochs=num_epochs,
      shuffle_buffer_size=int(1e6),
      sloppy=True)

    parse_function = self.load_data
    # The map function will call _parse_function on each (filename, label) pair
    dataset = dataset.map(parse_function).batch(self.config.batch_size, drop_remainder=False)
    
    iterator = dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    init_op = iterator.initializer
    return init_op, inputs, labels, num_minibatches

  def load_data(self, filename, label):
    dir_root = self.get_cwd()
    filename = filename['Path'][0]
    image_string = tf.read_file(dir_root + "/" + filename)
    image = tf.image.decode_png(image_string)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if not self.config.grayscale:
      image = tf.image.grayscale_to_rgb(image)
    if self.config.normalize_input:
      image = image - tf.constant(self.mean, dtype=tf.float32)
      image = image / tf.constant(self.std,dtype=tf.float32)

    # if self.config.augment and self.config.load_type == 'train' and self.config.model_name == 'mil':
    #   # For each channel (case)
    #   for i in range(image.get_shape()[2]):
    #     # Random flip
    #     im_slice = image[:,:,i]
    #     im_slice = tf.image.random_flip_left_right(im_slice)
    #     im_slice = tf.image.random_flip_up_down(im_slice)
    #     # Random rotation
    #     im_slice = tf.image.rot90(im_slice, int(random.random() * 4))
    #     # Random crop
    #     new_shape = tf.convert_to_tensor((int(299 * (1 - random.random() * .25)), int(299 * (1 - random.random() * .25)), 4))
    #     im_slice = tf.image.random_crop(im_slice, new_shape)
    #     im_slice = tf.image.resize_images(im_slice, (tf.convert_to_tensor((299,299))))
    #     # Random rebrighten
    #     # image = tf.image.random_brightness(image, max_delta=0.1)

    #     if i == 0:
    #       new_image = im_slice
    #     else:
    #       new_image = new_image.vstack(im_slice)
    #   image = new_image
    return image, label

  def get_cwd(self):
    return os.getcwd().replace('\\','/').replace('/src','/').replace('/util','/')
