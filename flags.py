import tensorflow as tf
import os

tf.app.flags.DEFINE_string('mode', 'train', 'train / val / test')
tf.app.flags.DEFINE_string('model_dir', 'model', 'model dir')
tf.app.flags.DEFINE_string('data_dir', 'data', 'data dir')

tf.app.flags.DEFINE_string('model_type', 'RNN', 'DNN / CNN / RNN')
tf.app.flags.DEFINE_integer('max_length', 300, 'max sentence length')
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_float('dp', 0.3, 'drop rate')

tf.app.flags.DEFINE_string('clean', False, 'clean models')
tf.app.flags.DEFINE_string('load', '', 'load model')
"""
tf.app.flags.DEFINE_integer('print_step', 1, 'printing step')
tf.app.flags.DEFINE_integer('save_step', 2, 'saving step')
tf.app.flags.DEFINE_integer('max_step', 4, 'number of steps')
"""

tf.app.flags.DEFINE_integer('print_step', 500, 'printing step')
tf.app.flags.DEFINE_integer('save_step', 5000, 'saving step')
tf.app.flags.DEFINE_integer('max_step', 20000, 'number of steps')

FLAGS = tf.app.flags.FLAGS
FLAGS.class_n = 6

if FLAGS.mode != 'train':
    FLAGS.batch_size = 5000

#FLAGS.units = [1024, 512]
FLAGS.units = [256]
FLAGS.filter = [16, 32, 64]
FLAGS.kernel = [3, 5]

FLAGS.model_type = FLAGS.model_type.upper()

FLAGS.model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
os.system('mkdir -p {}'.format(FLAGS.model_dir))

if FLAGS.clean:
  os.system('/bin/rm {}/*'.format(FLAGS.model_dir))

if FLAGS.load != '':
  FLAGS.load = os.path.join(FLAGS.model_dir, 'model-{}'.format(FLAGS.load))

