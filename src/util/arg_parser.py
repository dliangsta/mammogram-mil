import argparse
import sys
import tensorflow as tf

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--model_name',
      type=str,
      default="mil",
      help='Type of model: mil, transfer_mil, transfer, baseline',
      required=True
  )
  parser.add_argument(
      '--mil_type',
      type=str,
      default="vote",
      help='Type of MIL: stack, stitch, or vote'
  )
  parser.add_argument(
      '--vote_type',
      type=str,
      default="nn",
      help='Type of voting: nn, mean, max'
  )
  parser.add_argument(
      '--sigmoid_before_vote',
      action='store_true',
      help='Sigmoid the logits before voting as well.'
  )
  parser.add_argument(
      '--image_dir',
      type=str,
      default='../data/shuffle_within_case',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=2500,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=1e-3,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--positive_error_rate_multiplier',
      type=float,
      default=4,
      help='Multiplier for loss on positive examples'
  )
  parser.add_argument(
      '--regularization_scale',
      type=float,
      default=1e-7,
      help='Scale for L2 regularization'
  )
  parser.add_argument(
      '--regularization_loss_weight',
      type=float,
      default=1e-2,
      help='Scale for L2 regularization'
  )
  parser.add_argument(
      '--validate_freq',
      type=int,
      default=1,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--save_freq',
      type=int,
      default=20,
      help='How often to save the model weights'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--tfhub_module',
      type=str,
      default='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
      help=" Which TensorFlow Hub module to use."
  )
  parser.add_argument(
      '--image_height',
      type=int,
      default=299,
      help='Image height.'
  )
  parser.add_argument(
      '--image_width',
      type=int,
      default=299,
      help='Image width.'
  )
  parser.add_argument(
      '--image_channels',
      type=int,
      default=3,
      help='Number of image channels.'
  )
  parser.add_argument(
      '--cpu_test',
      action='store_true',
      help='Uses a very small amount of dummy data.'
  )
  parser.add_argument(
      '--debug',
      action='store_true',
      help='If true, print out debug info.'
  )
  parser.add_argument('--freeze', action='store_true')
  parser.add_argument('--grayscale', action='store_true')
  parser.add_argument('--normalize_input', action='store_true')
  parser.add_argument('--augment', action='store_true')
  parser.add_argument('--dropout', action='store_true')
  parser.add_argument('--batch_norm', action='store_true')
  parser.add_argument('--quick', action='store_true')
  parser.add_argument('--large', action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.trainable = not FLAGS.freeze
  return FLAGS, unparsed
