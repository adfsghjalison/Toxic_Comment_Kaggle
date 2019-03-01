from __future__ import print_function
from flags import FLAGS
import numpy as np
import tensorflow as tf
import pandas as pd
import os, csv, json
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

class utils():
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.data_dir = FLAGS.data_dir
        self.mode = FLAGS.mode
        self.model_type = FLAGS.model_type
        self.max_length = FLAGS.max_length
        self.class_n = FLAGS.class_n
        self.train_data = os.path.join(self.data_dir, 'train.csv')
        self.test_data = os.path.join(self.data_dir, 'test.csv')
        self.val_data = os.path.join(self.data_dir, 'try.csv')
        self.output = os.path.join(self.data_dir, 'prediction.csv')
        self.dict_data = os.path.join(self.data_dir, 'dict')
        self.word_embd_path = os.path.join(self.data_dir, 'wordvec')

        self.dict = self.get_dict()
        self.eos = self.dict['__EOS__']
        self.unk = self.dict['__UNK__']

        self.xv_size = len(self.dict)
        print('\nWord Size : {}'.format(self.xv_size))

        self.name = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        if self.mode != 'train':
            self.output = csv.writer(open(self.output, 'w'))
            self.output.writerow(self.name)

    def get_dict(self):
        return json.load(open(self.dict_data, 'r'))

    def get_x(self, sen):
        sen = sen.replace('"', '').replace('.', '')
        if self.model_type == 'DNN':
            return self.get_bow(sen)
        else:
            return self.get_ids(sen)

    def get_bow(self, sen):
      x = [0] * self.xv_size
      for w in sen.split():
        x[self.dict.get(w, self.unk)] += 1
      #w_total = max(sum(x), 1)
      #x = [float(i)/w_total for i in x]
      return x

    def get_ids(self, sen):
        x = [self.eos] * self.max_length
        for i, w in enumerate(sen.split()):
            if i >= self.max_length:
                break
            x[i] = self.dict.get(w, self.unk)
        return x

    def get_len(self, sen):
        return min(len(sen.split()), self.max_length)

    def load_word_embedding(self):
        embd = []
        with open(self.word_embd_path, 'r') as f:
            for line in f.readlines():
                row = line.strip().split(' ')
                embd.append(row[1:])
            print('\nWord Embedding loaded ... ')
            embedding = np.asarray(embd, 'f')
            return embedding

    def get_train_batch(self):
        df = pd.read_csv(self.train_data)
        num = len(df)
        print('\nData Number : {}'.format(num))
        while(True):
            df = df.sample(frac=1)
            begin = 0
            while begin < num:
              if begin + self.batch_size < num:
                end = begin + self.batch_size
              else:
                end = num
              sens = df.iloc[begin : end, 1]
              x = np.array([self.get_x(s) for s in sens])
              y = np.array(df.iloc[begin : end, 2:])
              l = np.array([self.get_len(s) for s in sens]) if self.model_type == 'RNN' else []
              yield x, y, l
              begin += self.batch_size

    def get_test_batch(self, mode='test'):
        if mode == 'test':
          df = pd.read_csv(self.test_data)
        else:
          df = pd.read_csv(self.val_data) 
        num = len(df.index)
        print('\nData Number : {}'.format(num))
        begin = 0
        l = []
        while begin < num:
          if begin + self.batch_size < num:
            end = begin + self.batch_size
          else:
            end = num
          id = df.iloc[begin : end, 0]
          sens = df.iloc[begin : end, 1]
          x = np.array([self.get_x(s) for s in sens])
          y = np.array(df.iloc[begin : end, 2:]) if mode == 'val' else np.zeros((end-begin, 6))
          l = np.array([self.get_len(s) for s in sens]) if self.model_type == 'RNN' else []
          yield id, x, y, l
          begin += self.batch_size

    def write_prediction(self, r):
        self.output.writerow(r)

    def print_val(self, step, acc, p, r, loss):
        p = round(p, 2)
        r = round(r, 2)
        acc = round(acc, 2)
        loss = round(loss, 3)
        print_info = [('Step', step), ('Accuray', acc), ('Precision', p), ('Recall', r), ('Loss', loss)]
        for i, j in print_info:
            print("{}: {:<5}  ".format(i, j), end='')
        print('')
        
def check_data():
  step = 0
  for x, y in utils().get_train_batch():
    step += 1
    print(x)
    print(y)
    if step > 2:
      break

ep = pow(10, -7)

def f_score(beta, ps, rs):
    ps = [i+ep for i in ps]
    rs = [i+ep for i in rs]
    return [round((1.0+beta*beta)*p*r/(beta*beta*p+r), 2) for p, r in zip(ps, rs)]

if __name__ == '__main__':
  check_data()

