import sys
import tensorflow as tf
tf.random.set_random_seed(42)

from mil_models import TransferMILModel, BaselineMILModel
from transfer_model import TransferModel
from baseline_model import BaselineModel
from util.arg_parser import parse_args
from copy import deepcopy

def main(_):
  eval_config = deepcopy(FLAGS)
  eval_config.image_dir = '../data/stitch_case_images/'
  eval_config.image_height = 299 * 2
  eval_config.image_width = 299 * 2
  eval_config.image_channels = 3
  eval_config.grayscale = True
  # Create model and hardcode arguments that are specific to models
  if FLAGS.model_name == 'mil':
    assert FLAGS.mil_type is not None
    if FLAGS.mil_type == 'stack':
      eval_config.image_dir = FLAGS.image_dir
      eval_config.image_height = FLAGS.image_height
      eval_config.image_width = FLAGS.image_width
      eval_config.grayscale = FLAGS.grayscale
      FLAGS.image_channels = 4
      eval_config.image_channels = 4
    else: # stitch and vote
      FLAGS.image_dir = '../data/stitch_case_images/'
      FLAGS.image_height = 299 * 2
      FLAGS.image_width = 299 * 2
      FLAGS.image_channels = 3
    FLAGS.grayscale = True
    # FLAGS.augment = True
    model = BaselineMILModel('mil', FLAGS, eval_config)
  elif FLAGS.model_name == 'transfer_mil':
    assert FLAGS.mil_type == 'vote'
    FLAGS.image_dir = '../data/stitch_case_images/'
    FLAGS.image_height = 299 * 2
    FLAGS.image_width = 299 * 2
    FLAGS.image_channels = 3
    FLAGS.grayscale = True
    FLAGS.augment = True
    model = TransferMILModel('mil', FLAGS, eval_config)
  elif FLAGS.model_name == 'transfer':
    model = TransferModel('transfer_inception', FLAGS, eval_config)
  elif FLAGS.model_name == 'baseline':
    FLAGS.learning_rate = 1e-7
    model = BaselineModel('baseline', FLAGS, eval_config)
  else:
    raise Exception("Unrecognized model name '{}'".format(FLAGS.model_name))


  model.train()
  model.test('test')

      
if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  print("Training {} model!".format(FLAGS.model_name))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
