import tensorflow as tf
import os

tf.app.flags.DEFINE_string('mode', 'train', 'train / test')
tf.app.flags.DEFINE_string('model_dir', 'model', 'model dir')
tf.app.flags.DEFINE_string('data_dir', 'data', 'data dir')

tf.app.flags.DEFINE_string('model_type', 'dnn', 'dnn')
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_float('dp', 0.3, 'drop rate')

tf.app.flags.DEFINE_string('load', '', 'load model')

tf.app.flags.DEFINE_integer('print_step', 100, 'printing step')
tf.app.flags.DEFINE_integer('save_step', 1000, 'saving step')
tf.app.flags.DEFINE_integer('max_step', 10000, 'number of steps')

FLAGS = tf.app.flags.FLAGS

FLAGS.units = [1024, 512]
#FLAGS.filter = [32, 64, 128]
#FLAGS.kernel = [3, 3]

if FLAGS.load != '':
  FLAGS.load = os.path.join(FLAGS.model_dir, 'model-{}'.format(FLAGS.load))

