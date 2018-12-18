"""
Adapted from https://github.com/keras-team/keras/issues/4465.
This is a shallow fully connected  Keras model.
"""

import os
import sys
import cv2
import csv
import numpy as np
import sklearn.metrics as sklm
import tensorflow as tf

from util.arg_parser import parse_args
from util.load_data import load_data

from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.models import model_from_json

from keras import backend as K
img_dim_ordering = 'tf'
K.set_image_dim_ordering(img_dim_ordering)

# two fc layers
def create_shallow_model(img_shape, num_classes, activation_fn):
  # create your own input format
  keras_input = Input(shape=img_shape, name = 'image_input')
  
  # add the fully-connected layers 
  x = Flatten(name='flatten')(keras_input)
  x = Dense(4096, activation=activation_fn, name='fc1')(x)
  x = Dense(2048, activation=activation_fn, name='fc2')(x)
  x = Dense(num_classes, activation='softmax', name='predictions')(x)
  
  # create your own model 
  pretrained_model = Model(inputs=keras_input, outputs=x)
  pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  return pretrained_model

def main(_):
  # loading the data
  x_train, y_train = load_data('train', FLAGS.image_dir, (FLAGS.image_height, FLAGS.image_width), FLAGS.train_data_keep_prob)
  x_dev, y_dev = load_data('dev', FLAGS.image_dir, (FLAGS.image_height, FLAGS.image_width), FLAGS.train_data_keep_prob)
  x_test, y_test = load_data('test', FLAGS.image_dir, (FLAGS.image_height, FLAGS.image_width), FLAGS.train_data_keep_prob)

  # train the model or load the existing model
  if (FLAGS.use_pretrained and os.path.exists(FLAGS.model_name + ".json")):
    json_file = open(FLAGS.model_name + ".json", 'r')
    # pretrained model
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # pretrained weights
    model.load_weights(FLAGS.model_name + ".h5")
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Loaded pretrained model")
  else:
    # train
    model = create_shallow_model(x_train.shape[1:], y_train.shape[0], 'relu')
    hist = model.fit(x_train, y_train, epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, validation_data=(x_dev, y_dev), verbose=1)

    # save model and weights
    model_json = model.to_json()
    with open (FLAGS.model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(FLAGS.model_name + ".h5")
    print("Saved baseline model to disk")

  # evaluate model
  score = model.evaluate(x_test, y_test, verbose=1)
  print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  # Have to make images small to fit in memory
  FLAGS.image_height = 64
  FLAGS.image_width = 64
  FLAGS.batch_size = 10
  FLAGS.use_pretrained = False
  FLAGS.model_name = 'baseline_model'
  FLAGS.image_dir = "../data/group_by_case"
  FLAGS.num_epochs = 30
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
