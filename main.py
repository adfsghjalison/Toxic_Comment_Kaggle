from flags import FLAGS
from model import Model
import time

def run():
  model = Model(FLAGS)
  t_start = time.clock()
  if FLAGS.mode == 'train':
    model.train()
  elif FLAGS.mode == 'test' or FLAGS.mode == 'val':
    model.test(FLAGS.mode)
  t_exe = int(time.clock() - t_start)
  print "\nTime :  {}:{}\n".format(t_exe/60, t_exe%60)

if __name__ == '__main__':
  run()

