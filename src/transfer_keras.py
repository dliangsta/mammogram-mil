# adapted from https://github.com/keras-team/keras/issues/4465
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

# the model
def create_model(img_shape, num_classes, activation_fn):
  # model_conv = InceptionV3(weights=None, include_top=False)
  model_conv = InceptionV3(weights='imagenet', include_top=False)
  # freeze layers, remove this loop to train the whole network
  for layer in model_conv.layers:
      layer.trainable = False
  # print model_conv.summary()
  
  # create your own input format
  keras_input = Input(shape=img_shape, name = 'image_input')
  
  # use the generated model 
  output_conv = model_conv(keras_input)
  
  # add the fully-connected layers 
  x = Flatten(name='flatten')(output_conv)
  x = Dense(2048, activation=activation_fn, name='fc1')(x)
  x = Dense(1024, activation=activation_fn, name='fc2')(x)
  # kept this 10 neuron dense layer because otherwise it wouldn't do anything
  # x = Dense(10, activation=activation_fn, name='fc_out')(x)
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
    model = create_model(x_train.shape[1:], y_train.shape[0], 'relu')
    hist = model.fit(x_train, y_train, epochs=10, batch_size=FLAGS.batch_size, validation_data=(x_dev, y_dev), verbose=1)

    # save model and weights
    model_json = model.to_json()
    with open (FLAGS.model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(FLAGS.model_name + ".h5")
    print("Saved baseline model to disk")

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
  FLAGS.image_height = 256
  FLAGS.image_width = 256
  FLAGS.batch_size = 10
  FLAGS.use_pretrained = False
  FLAGS.model_name = 'baseline_model'
  FLAGS.image_dir = "../data/group_by_case"
  FLAGS.num_epochs = 30
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
