import os
import sys
import glob
import random
import skimage as sk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from pprint import PrettyPrinter
from util.data_iterator import DataIterator


class Model:
  """
  To define a model, inherit this class and implement one method:
    get_probabilities(self, inputs, reuse).

  Optionally, you can change the train graph, eval graph, or loss function by
    overriding the respective methods.
  """
  def __init__(self, model_name, config, eval_config):
    self.model_name = model_name
    self.config = config
    self.prepare_logs()
    self.sess = tf.Session()
    self.train_data_iterator = DataIterator(config)
    self.eval_data_iterator = DataIterator(eval_config)

    # Input and training status placeholders
    self.train_inputs = tf.placeholder(tf.float32, [None, self.config.image_height, self.config.image_width, self.config.image_channels])
    self.eval_inputs = tf.placeholder(tf.float32, [None, eval_config.image_height, eval_config.image_width, eval_config.image_channels])
    self.label_input = tf.placeholder(tf.float32, [None, 1])
    self.training = tf.placeholder(tf.bool)

    print("---Setting up training graph---")
    print(self.train_inputs)
    self.set_up_train_graph()
    print("---Setting up eval graph---")
    print(self.eval_inputs)
    self.set_up_eval_graph()
    self.set_up_loss()
    self.create_optimizer()
    self.saver = tf.train.Saver()

  def prepare_logs(self):
    self.experiment_num = len(glob.glob('../experiments/experiment_*/'))
    print(self.experiment_num)
    self.root_dir = '../experiments/experiment_{}/'.format(self.experiment_num)
    if not os.path.isdir("../experiments/"):
      os.makedirs("../experiments/")
    self.log_file = self.root_dir + 'log.txt'
    assert not os.path.isdir(self.root_dir)
    os.makedirs(self.root_dir)
    os.makedirs(self.root_dir + "plots/")
    os.makedirs(self.root_dir + "model/")

    pp = PrettyPrinter()
    self.log(pp.pformat(self.config.__dict__))
    self.log(pp.pformat(self.__dict__))

    with open('../experiments/experiments.txt','a') as f:
      f.write(self.root_dir + '\n')
      f.write(pp.pformat(self.config.__dict__) + '\n')
      f.write(pp.pformat(self.__dict__) + '\n\n')

  def get_probabilities(self, inputs, reuse=False):
    """
    Returns tensor of probabilities [batch_size, 1]
    """
    raise Exception("Implement this.")

  def set_up_train_graph(self):
    """Optionally override if you want your model to train in a different way."""
    self.train_probabilities = self.get_probabilities(self.train_inputs)
    self.train_predictions = tf.round(self.train_probabilities)
    self.train_precision, self.train_recall, self.train_f1, self.train_accuracy = self.create_metrics(self.train_predictions)

  def set_up_eval_graph(self):
    """Optionally override if you want your model to evaluate in a different way."""
    left_images, right_images = tf.split(self.eval_inputs, 2, axis=1)
    left_cc, left_mlo = tf.split(left_images, 2, axis=2)
    right_cc, right_mlo = tf.split(right_images, 2, axis=2)
    eval_logits = []
    for eval_image in [left_cc, left_mlo, right_cc, right_mlo]:
      eval_logits.append(self.get_probabilities(eval_image, reuse=True))
      print("~~~")

    stacked_eval_logits = tf.stack(eval_logits, axis=1)
    eval_vote = tf.reduce_max(stacked_eval_logits, axis=1, keepdims=True)
    self.eval_probabilities = tf.nn.sigmoid(eval_vote)
    self.eval_predictions = tf.round(self.eval_probabilities)
    self.eval_precision, self.eval_recall, self.eval_f1, self.eval_accuracy = self.create_metrics(self.eval_predictions)

  def set_up_loss(self):
    """Optionally override if you want your model to use a different loss."""
    self.loss = -tf.reduce_mean(self.config.positive_error_rate_multiplier * self.label_input * tf.log(self.train_probabilities) + (1 - self.label_input) * tf.log(1 - self.train_probabilities))
    self.loss += self.config.regularization_loss_weight * tf.reduce_mean(tf.losses.get_regularization_losses())

  def create_conv_layers(self, l, conv_layers):
    for filters, kernel_size, kernel_stride, pool_stride in conv_layers:
      l = tf.layers.conv2d(inputs=l, 
                           filters=filters, 
                           kernel_size=[kernel_size] * 2, 
                           strides=[kernel_stride]*2, 
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.regularization_scale),
                           padding="valid")
      print(l.shape)
      if self.config.batch_norm:
        # Apply batch norm
        l = tf.contrib.layers.batch_norm(l)

      # Apply non-linear activation
      l = tf.nn.relu(l)
      
      # if self.config.dropout:
      #   # Apply dropout
      #   l = tf.layers.dropout(l, training=self.training)

      # Pool
      l = tf.layers.max_pooling2d(inputs=l, pool_size=[pool_stride] * 2, strides=pool_stride)
      print(l.shape)
    
    return l

  def create_dense_layers(self, l, dense_layers):
    for units in dense_layers:
      l = tf.layers.dense(inputs=l, 
                          units=units, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.regularization_scale))
      print(l.shape)
      if self.config.batch_norm:
        # Apply batch norm
        l = tf.contrib.layers.batch_norm(l)

      # Apply non-linear activation
      l = tf.nn.relu(l)

      if self.config.dropout:
        # Apply dropout
        l = tf.layers.dropout(l, training=self.training)

    # Final output layer
    l = tf.layers.dense(inputs=l,
                        units=1,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.regularization_scale))
    return l

  def create_metrics(self, predictions):
    """
    Needs to be called after creating a model.
    """
    self.train_metrics = None
    self.dev_metrics = None
    # Metrics
    true_positive = tf.cast(tf.count_nonzero(predictions * self.label_input), tf.float32)
    true_negative = tf.cast(tf.count_nonzero((predictions - 1) * (self.label_input - 1)), tf.float32)
    false_positive = tf.cast(tf.count_nonzero(predictions * (self.label_input - 1)), tf.float32)
    false_negative = tf.cast(tf.count_nonzero((predictions - 1) * self.label_input), tf.float32)

    tiny_number = 1e-10
    precision = true_positive / (true_positive + false_positive + tiny_number)
    recall = true_positive / (true_positive + false_negative + tiny_number)
    f1 = 2 * precision * recall / (precision + recall + tiny_number)

    accuracy = tf.reduce_mean(tf.cast(tf.math.equal(predictions, self.label_input), tf.float32))

    return precision, recall, f1, accuracy

  def create_optimizer(self):
    """
    Needs to be called after creating a model.
    """
    num_minibatches = self.train_data_iterator.get_ops('train')[3]

    self.learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate, 
                                          global_step=tf.train.get_or_create_global_step(), 
                                          decay_rate = .96,
                                          # every 10 epochs
                                          decay_steps = num_minibatches * 5)

    # Train step
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.step = self.optimizer.minimize(
        loss=self.loss,
        global_step=tf.train.get_global_step())

  def augment_images(self, images):
    # For each image in images
    for i in range(images.shape[0]):
      image = images[i,...]
      if self.config.model_name == 'mil' and self.config.mil_type == 'stack':
        # For each channel (case image)
        for j in range(images.shape[3]):
          # Save augmented image
          images[i,...,j] = self.augment_image(images[i,...,j])
      else:
        images[i,...] = self.augment_image(images[i,...])
    return images

  def augment_image(self, image):
    image = sk.transform.rotate(image, 90 * random.randint(0,3))
    # Random flip
    if random.random() < .5:
      image = np.flipud(image)
    if random.random() < .5:
      image = np.fliplr(image)
    return image

  def train(self):
    self.sess.run(tf.global_variables_initializer())
    init_op, inputs_op, labels_op, num_minibatches = self.train_data_iterator.get_ops('train')

    if self.config.quick:
      num_minibatches = num_minibatches // 100

    # test against dev set
    train_metrics = self.test('train')
    dev_metrics = self.test('dev')
    self.log("Iteration {}\n"
          "Train metrics\tAverage precision: {}\tAverage recall: {}\tAverage F1: {}\tAverage accuracy: {}\n"
          "Dev metrics\tAverage precision: {}\tAverage recall: {}\tAverage F1: {}\tAverage accuracy: {}".format(
            0, 
            *train_metrics.tolist(),
            *dev_metrics.tolist()))

    # Save metrics
    self.train_metrics = train_metrics
    self.dev_metrics = dev_metrics

    # Train
    for epoch in range(1, self.config.num_epochs + 1):
      self.sess.run(init_op)
      t = trange(num_minibatches)
      t.set_description("Train accuracy: ")
      try:
        # Train over minibatches
        for i in t:
          # Get data from iterator
          inputs, labels = self.sess.run([inputs_op, labels_op])
          # Augment if required
          if self.config.augment:
            inputs = self.augment_images(inputs)
          # Train and get metrics
          _, predictions, precision, recall, f1, accuracy = self.sess.run(
            [self.step, self.train_predictions, self.train_precision, self.train_recall, self.train_f1, self.train_accuracy],
            feed_dict={self.train_inputs: inputs, self.label_input: labels, self.training: True})
          
          # Store metrics
          if i == 0:
            metrics = np.array([precision, recall, f1, accuracy])
          else:
            metrics = np.vstack((metrics, [precision, recall, f1, accuracy]))
            t.set_description("Train accuracy: %.5f" % np.mean(metrics[:,3]))
            t.refresh()

          if i % 100 and self.config.debug:
            print("Train inputs: {}".format(inputs))
            print("Train labels: {}".format(labels))
            print("Train predictions: {}".format(predictions))
      except tf.errors.OutOfRangeError:
        # Okay to come here, as this is TF expected behavior
        pass

      # Collect train metrics
      train_metrics = np.mean(metrics, axis=0)

      # Periodically test against dev set
      if (epoch % self.config.validate_freq) == 0 or (epoch == self.config.num_epochs):
        dev_metrics = self.test('dev')
        self.log("Iteration {}\n"
              "Train metrics\tAverage precision: {}\tAverage recall: {}\tAverage F1: {}\tAverage accuracy: {}\n"
              "Dev metrics\tAverage precision: {}\tAverage recall: {}\tAverage F1: {}\tAverage accuracy: {}".format(
                epoch, 
                *train_metrics.tolist(),
                *dev_metrics.tolist()))
        if epoch > 2:
          self.write_metrics()

      if (epoch % self.config.save_freq) == 0:
        self.saver.save(self.sess, self.root_dir + "model/model", global_step=tf.train.get_global_step())

      # Save metrics
      self.train_metrics = np.vstack((self.train_metrics, train_metrics))
      self.dev_metrics = np.vstack((self.dev_metrics, dev_metrics))

  def test(self, split):
    assert split in ('train', 'dev', 'test')
    if split == 'train':
      inputs_placeholder = self.train_inputs
      data_iterator = self.train_data_iterator
      predictions_op = self.train_predictions
      precision_op = self.train_precision
      recall_op = self.train_recall
      f1_op = self.train_f1
      accuracy_op = self.train_accuracy
    else:
      inputs_placeholder = self.eval_inputs
      data_iterator = self.eval_data_iterator
      predictions_op = self.eval_predictions
      precision_op = self.eval_precision
      recall_op = self.eval_recall
      f1_op = self.eval_f1
      accuracy_op = self.eval_accuracy
    init_op, inputs_op, labels_op, num_minibatches = data_iterator.get_ops(split)

    self.sess.run(init_op)

    for i in range(num_minibatches):
      inputs, labels = self.sess.run([inputs_op, labels_op])
      predictions, precision, recall, f1, accuracy = self.sess.run(
        [predictions_op, precision_op, recall_op, f1_op, accuracy_op],
        feed_dict={inputs_placeholder: inputs, self.label_input: labels, self.training: False})
      if i == 0:
        metrics = np.array([precision, recall, f1, accuracy])
      else:
        metrics = np.vstack((metrics, [precision, recall, f1, accuracy]))

      if i % 100 and self.config.debug:
        print("{} inputs: {}".format(split.capitalize(), inputs))
        print("{} labels: {}".format(split.capitalize(), labels))
        print("{} predictions: {}".format(split.capitalize(), predictions))

    metrics = np.mean(metrics, axis=0)


    if split == 'test':
       self.log("Test metrics\tAverage precision: {}\tAverage recall: {}\tAverage F1: {}\tAverage accuracy: {}".format(
                *metrics.tolist()))

    return metrics

  def write_metrics(self):
    # Plots for all metrics for each of train and dev
    metrics = ['precision','recall','f1','accuracy']
    for stats, split in zip([self.train_metrics, self.dev_metrics], ["train","dev"]):
      colors = ['r','g','b','c']
      for i, (color, metric) in enumerate(zip(colors,metrics)):
        plt.plot(range(len(stats)), stats[:,i], color, label=metric)

      plt.xlabel("Epochs")
      plt.title("{} metrics over time".format(split.capitalize()))
      plt.legend(loc='upper left')
      if not os.path.isdir('plots/'):
        os.makedirs('plots/')
      plt.savefig(self.root_dir + "plots/{}_metrics.png".format(split))
      plt.clf()
    
    # Plots for train and dev of each metric
    for i, metric in enumerate(metrics):
      plt.plot(range(len(self.train_metrics)), self.train_metrics[:,i], 'r', label='train')
      plt.plot(range(len(self.dev_metrics)), self.dev_metrics[:,i], 'g', label='dev')

      plt.xlabel("Epochs")
      plt.ylabel(metric.capitalize())
      plt.title("{} over time".format(metric.capitalize()))
      plt.legend(loc='upper left')
      if not os.path.isdir('plots/'):
        os.makedirs('plots/')
      plt.savefig(self.root_dir + "plots/{}.png".format(metric))
      plt.clf()

    # Update central experiments.csv
    new_line = ','.join(['{}'] * 9).format(self.experiment_num, *self.train_metrics[-1,...].tolist(), *self.dev_metrics[-1,...].tolist())
    lines = ['experiment_num,train_precision,train_recall,train_f1,train_accuracy,dev_precision,dev_recall,dev_f1,dev_accuracy', new_line]
    found = False
    experiments_csv_path = '../experiments/experiments.csv'
    if os.path.isfile(experiments_csv_path):
      with open(experiments_csv_path,'r') as f:
        lines = [line.strip() for line in f.readlines()]
        for i, line in enumerate(lines):
          if line.split(',')[0] == str(self.experiment_num):
            lines[i] = new_line
            found = True
            break
      if not found:
        lines.append(new_line)
        
    with open(experiments_csv_path,'w') as f:
      for line in lines:
        f.write(line + '\n')
            

  def log(self, text):
    print(text)
    with open(self.log_file,'a') as f:
      if text[-1] != '\n':
        text += '\n'
      f.write(text)
