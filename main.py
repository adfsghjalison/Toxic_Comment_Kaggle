from flags import FLAGS
from model import Model


def run():
  model = Model(FLAGS)
  if FLAGS.mode == 'train':
    model.train()
  elif FLAGS.mode == 'test':
    model.test()
  elif FLAGS.mode == 'val':
    model.val()

if __name__ == '__main__':
  run()

