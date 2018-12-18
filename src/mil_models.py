import sys
import tensorflow as tf
import tensorflow_hub as hub

from model import Model

class MILModel(Model):

  def get_probabilities(self, input, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
      if self.config.mil_type == "vote":
        left_images, right_images = tf.split(input, 2, axis=1)
        left_cc, left_mlo = tf.split(left_images, 2, axis=2)
        right_cc, right_mlo = tf.split(right_images, 2, axis=2)
        inputs = [left_cc, left_mlo, right_cc, right_mlo]
      else:
        inputs = [input]

      logits = []
      for input in inputs:
        print("~~~")
        print(input.shape)

        l = self.encode(input)
        print(l.shape)

        # Flatten output of previous layer, then feed into dense
        l = tf.layers.flatten(l)
        print(l.shape)

        if self.config.mil_type == "vote" and self.config.sigmoid_before_vote:
          l = tf.sigmoid(l)

        logits.append(l)

      print("---")
      if self.config.mil_type == "vote":
        stacked_logits = tf.stack(logits, axis=1)
        print(stacked_logits.shape)
        if self.config.vote_type == "nn":
          print("nn")
          # Hack to get rid of the last dimension while keeping the shape known.
          stacked_logits = tf.reduce_mean(stacked_logits, axis=2)
          print(stacked_logits.shape)
          l = self.create_dense_layers(stacked_logits, [len(logits)])
        elif self.config.vote_type == "mean":
          print("mean")
          l = tf.reduce_mean(stacked_logits, axis=1, keepdims=True)
        elif self.config.vote_type == "max":
          print("max")
          l = tf.reduce_max(stacked_logits, axis=1, keepdims=True)
      else:
        l = logits[0]
      print(l.shape)

      return tf.nn.sigmoid(l)

  def set_up_eval_graph(self):
    """Override the default eval graph to do the proper type of MIL."""
    self.eval_probabilities = self.get_probabilities(self.eval_inputs, reuse=True)
    self.eval_predictions = tf.round(self.eval_probabilities)
    self.eval_precision, self.eval_recall, self.eval_f1, self.eval_accuracy = self.create_metrics(self.eval_predictions)

class BaselineMILModel(MILModel):

  def encode(self, input):
      
    if self.config.large:
      conv_layers = [(32, 7, 2, 2), (64, 5, 2, 2), (128, 3, 2, 2), (256, 3, 1, 2)]
    else:
      conv_layers = [(32, 5, 1, 3), (64, 5, 1, 3), (128, 3, 1, 2), (256, 3, 1, 2)]

    l = self.create_conv_layers(input, conv_layers)

    # Flatten output of previous layer, then feed into dense
    l = tf.layers.flatten(l)
    print(l.shape)

    dense_layers = [4096, 128]
    l = self.create_dense_layers(l, dense_layers)
    return l

class TransferMILModel(MILModel):
  def __init__(self, model_name, config):
    module_spec = hub.load_module_spec(config.tfhub_module)
    if config.trainable:
      self.module = hub.Module(module_spec, trainable=True, tags=['train'])
    else:
      self.module = hub.Module(module_spec, trainable=False)

    super(TransferMILModel, self).__init__(model_name, config)

    if config.trainable:
      self.log("Inception module is trainable!")
    else:
      self.log("Inception module is frozen!")

  def encode(self, input):

    l = self.module(input)

    # Flatten module output and add dense layers
    l = tf.layers.flatten(l)
    
    self.training = tf.placeholder(tf.bool)
    dense_layers = [128]
    l = self.create_dense_layers(l, dense_layers)
    return l