import sys
import tensorflow as tf

from model import Model


class BaselineModel(Model):

  def get_probabilities(self, inputs, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
      print(inputs.shape)

      # conv_layers = [(32, 5, 2, 2), (64, 5, 2, 2), (128, 3, 1, 2), (256, 3, 1, 2)]
      conv_layers = [(32, 5, 2, 2), (64, 5, 2, 2), (128, 3, 1, 2)]
      l = self.create_conv_layers(inputs, conv_layers)
      print(l.shape)


      # Flatten output of previous layer, then feed into dense
      l = tf.layers.flatten(l)
      print(l.shape)

      dense_layers = [2048, 512]
      l = self.create_dense_layers(l, dense_layers)
      print(l.shape)

      # Network output
      return tf.nn.sigmoid(l)
