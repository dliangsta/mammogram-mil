import sys
import tensorflow as tf
import tensorflow_hub as hub

from model import Model

class TransferModel(Model):

  def __init__(self, model_name, config, eval_config):
    module_spec = hub.load_module_spec(config.tfhub_module)
    if config.trainable:
      self.module = hub.Module(module_spec, trainable=True, tags=['train'])
    else:
      self.module = hub.Module(module_spec, trainable=False)

    super(TransferModel, self).__init__(model_name, config, eval_config) 
    if config.trainable:
      self.log("Inception module is trainable!")
    else:
      self.log("Inception module is frozen!")

  def get_probabilities(self, inputs, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
      print(inputs.shape)

      l = self.module(inputs)
      print(l.shape)

      # Flatten module output and add dense layers
      l = tf.layers.flatten(l)
      print(l.shape)
      
      dense_layers = [128]
      l = self.create_dense_layers(l, dense_layers)
      print(l.shape)

      return tf.nn.sigmoid(l)
