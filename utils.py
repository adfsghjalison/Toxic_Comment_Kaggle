from flags import FLAGS
import numpy as np
import tensorflow as tf
import pandas as pd
import os, csv, json

class utils():
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.data_dir = FLAGS.data_dir
        self.train_data = os.path.join(self.data_dir, 'train.csv')
        self.test_data = os.path.join(self.data_dir, 'test.csv')
        self.val_data = os.path.join(self.data_dir, 'train.csv')
        self.dict_data = os.path.join(self.data_dir, 'word.json')
        self.dict = self.get_dict()
        self.eos = self.dict['__EOS__']
        self.unk = self.dict['__UNK__']
        self.xv_size = len(self.dict)

    def get_dict(self):
        return json.load(open(self.dict_data, 'r'))

    def get_bow(self, sen):
      x = [0] * self.xv_size
      sen = sen.replace('.', '').replace('"', '')
      for w in sen.split():
        x[self.dict.get(w, self.unk)] = 1
      return x

    def get_train_batch(self):
        df = pd.read_csv(self.train_data)
        num = len(df.index)
        print num
        while(True):
            df = df.sample(frac=1)
            #print df.head()
            begin = 0
            while begin < num:
              if begin + self.batch_size < num:
                end = begin + self.batch_size
              else:
                end = num
              sens = df.iloc[begin : end, 1]
              x = np.array([self.get_bow(s) for s in sens])
              y = np.array(df.iloc[begin : end, 2:])
              yield x, y
              begin += self.batch_size

    def get_test_batch(self, mode='test'):
        if mode == 'test':
          df = pd.read_csv(self.test_data)
        else:
          df = pd.read_csv(self.val_data) 
        num = len(df.index)
        begin = 0
        while begin < num:
          if begin + self.batch_size < num:
            end = begin + self.batch_size
          else:
            end = num
          ids = df.iloc[begin : end, 0].tolist()
          sens = df.iloc[begin : end, 1]
          x = np.array([self.get_bow(s) for s in sens])
          if mode == 'test':
            yield ids, x
          else:
            y = np.array(df.iloc[begin : end, 2:])
            yield x, y
          begin += self.batch_size

def write_test(pre):
    cf = csv.writer(open(os.path.join(FLAGS.data_dir, 'prediction.csv'), 'w'))
    cf.writerow(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    for r in pre:
      cf.writerow(r)

def check_data():
  step = 0
  for x, y in utils().get_train_batch():
    step += 1
    if step > 100:
      break

if __name__ == '__main__':
  check_data()

