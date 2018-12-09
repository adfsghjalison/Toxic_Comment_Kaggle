import tensorflow as tf
import numpy as np
from utils import utils, write_test
import os, csv

class Model():
    def __init__(self, args):
        self.mode = args.mode
        self.model_dir = args.model_dir
        self.model_path = os.path.join(self.model_dir, 'model')
        self.data_dir = args.data_dir
        self.model_type = args.model_type
        self.units = args.units
        self.filter = args.filter
        self.kernel = args.kernel
        self.load = args.load
        self.print_step = args.print_step
        self.save_step = args.save_step
        self.max_step = args.max_step

        self.utils = utils()
        self.xv_size = self.utils.xv_size
        self.dp = args.dp

        self.sess = tf.Session()
        self.build(self.model_type)
        self.saver = tf.train.Saver(max_to_keep = 10)

    def build(self, model_type):
        self.x = tf.placeholder(tf.float32, shape=[None, self.xv_size])
        self.y = tf.placeholder(tf.float32, shape=[None, 6])

        if model_type == 'dnn':
          dense = tf.layers.dense(self.x, self.units[0], activation=tf.nn.sigmoid)
          z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')
          for i in range(1, len(self.units)):
            dense = tf.layers.dense(z, self.units[i], activation=tf.nn.sigmoid)
            z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')
        elif model_type == 'cnn':
          final_map_size = 28
          for i in range(len(self.filter)):
            final_map_size /= 2
          x_cnn = tf.reshape(self.x, [-1, 28, 28, 1])
          conv = tf.layers.conv2d(x_cnn, filters=self.filter[0], kernel_size = self.kernel, padding='same', activation=tf.nn.relu)
          pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2)
          for i in range(1, len(self.filter)):
            conv = tf.layers.conv2d(pool, filters=self.filter[i], kernel_size = self.kernel, padding='same', activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2)
          pool = tf.reshape(pool, [-1, final_map_size * final_map_size * self.filter[-1]])
          dense = tf.layers.dense(pool, 1024, activation=tf.nn.relu)
          z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')
          
        self.logits = tf.layers.dense(z, 6)

        # train
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        #self.op = tf.train.MomentumOptimizer(0.001, 0.8).minimize(self.loss)

        # test
        self.pred = tf.nn.sigmoid(self.logits)
        #print self.pred.shape
        #print self.y.shape
        self.auc = tf.metrics.auc(self.y, self.pred)
        self.pred_int = tf.cast(tf.greater(self.pred, 0.5), tf.float32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred_int, self.y), tf.float32))

    def train(self):

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        step = 1
        auc = [0.0] * 6
        acc = 0.0
        loss = 0.0

        if not os.path.exists(self.model_dir):
            os.system("mkdir -p {}".format(self.model_dir))
        
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))
            step = int(ckpt.model_checkpoint_path.split('-')[-1])

        for x, y in self.utils.get_train_batch():
            feed_dict = {self.x : x, self.y : y}
            output = [self.op, self.acc, self.auc, self.loss]
            _, acc_temp, auc_temp, loss_temp = self.sess.run(output, feed_dict)

            auc = [i+j for i, j in zip(auc, auc_temp)]
            acc += acc_temp
            loss += loss_temp

            if step % self.print_step == 0:
                auc = [i/self.print_step for i in auc]
                acc /= self.print_step
                loss /= self.print_step
                print("Step : {}    Acc : {}    Loss : {}".format(step, acc, loss))
                auc = [0.0] * 6
                acc = 0.0
                loss = 0.0

            if step % self.save_step == 0:
                print("Saving model ...")
                self.saver.save(self.sess, self.model_path, global_step=step)

            if step >= self.max_step:
                break

            step += 1

    def test(self):
        if self.load != '':
            self.saver.restore(self.sess, self.load)
            print("load model from {} ...".format(self.load))
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))

        self.sess.run(tf.local_variables_initializer())
        pre = []

        def score(k):
          k = round(k, 2)
          return k

        cf = csv.writer(open(os.path.join(self.data_dir, 'prediction.csv'), 'w'))
        cf.writerow(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
        for ids, x in self.utils.get_test_batch():
            feed_dict = {self.x : x}
            pred = self.sess.run([self.pred], feed_dict)[0]
            for i in range(len(ids)):
                p = [score(j) for j in pred[i]]
                p.insert(0, ids[i])
                cf.writerow(p)

    def val(self):
        if self.load != '':
            self.saver.restore(self.sess, self.load)
            print("load model from {} ...".format(self.load))
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))

        self.sess.run(tf.local_variables_initializer())
        acc = 0.0
        cnt = 0

        for x, y in self.utils.get_test_batch(mode='val'):
            feed_dict = {self.x : x, self.y : y}
            acc_temp = self.sess.run([self.acc], feed_dict)
            cnt += len(x)
            acc += acc_temp[0] * len(x)
        acc /= float(cnt)
        print("Auc : {}".format(auc))

