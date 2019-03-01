import tensorflow as tf
import numpy as np
from utils import utils
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
        self.word_embedding_dim = 300
        self.max_length = args.max_length

        self.utils = utils()
        self.dp = args.dp
        self.xv_size = self.utils.xv_size
        self.class_n = args.class_n
         
        self.sess = tf.Session()
        self.build(self.model_type)
        self.saver = tf.train.Saver(max_to_keep = 5)


    def build(self, model_type):

        self.y = tf.placeholder(tf.float32, shape=[None, self.class_n])
        self.activation = tf.nn.sigmoid

        if model_type == 'DNN':
            self.build_DNN()
        elif model_type == 'CNN':
            self.build_CNN()
        elif model_type == 'RNN':
            self.build_RNN()

    def build_DNN(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.xv_size])

        dense = tf.layers.dense(self.x, self.units[0], activation=self.activation)
        z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')
        for i in range(1, len(self.units)):
            dense = tf.layers.dense(z, self.units[i], activation=self.activation)
            z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')

        self.output_layer(z)
    
    def build_CNN(self):   
        self.x = tf.placeholder(tf.int64, [None, self.max_length])
        self.get_word_embedding()
        self.word_embedding = tf.expand_dims(self.word_embedding, -1)

        final_map_w = self.max_length
        final_map_h = self.word_embedding_dim
        for i in range(len(self.filter)):
          final_map_w /= 2
          final_map_h /= 2

        conv = tf.layers.conv2d(self.word_embedding, filters=self.filter[0], kernel_size = self.kernel, padding='same', activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2)
        for i in range(1, len(self.filter)):
          conv = tf.layers.conv2d(pool, filters=self.filter[i], kernel_size = self.kernel, padding='same', activation=tf.nn.relu)
          pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2)
        pool = tf.reshape(pool, [-1, final_map_w * final_map_h * self.filter[-1]])
        dense = tf.layers.dense(pool, 1024, activation=self.activation)
        z = tf.layers.dropout(dense, rate=self.dp, training=self.mode=='train')

        self.output_layer(z)

    def build_RNN(self):
        self.x = tf.placeholder(tf.int64, [None, self.max_length])
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.get_word_embedding()
        gru = tf.contrib.rnn.GRUCell(self.units[0])

        _, hidden_state = tf.nn.dynamic_rnn(gru, self.word_embedding, sequence_length=self.seq_len, dtype=tf.float32)

        self.output_layer(hidden_state)
        

    def get_word_embedding(self):
        self.embedding_pretrain = tf.placeholder(dtype=tf.float32, shape=[self.xv_size-2, self.word_embedding_dim])
        init = tf.contrib.layers.xavier_initializer()

        word_vector_EOS_UNK = tf.get_variable(
                name="word_vector_EOS_UNK",
                shape=[2, self.word_embedding_dim],
                initializer = init,
                trainable = True)

        pretrained_word_embd  = tf.get_variable(
                name="pretrained_word_embd",
                shape=[self.xv_size-2, self.word_embedding_dim],
                initializer = init,
                trainable = True)
        self.embd_init = pretrained_word_embd.assign(self.embedding_pretrain)
        
        word_embedding_matrix = tf.concat([word_vector_EOS_UNK, pretrained_word_embd], 0)
        self.word_embedding = tf.nn.embedding_lookup(word_embedding_matrix, self.x)
 
    def output_layer(self, z):
        self.logits = tf.layers.dense(z, self.class_n)
        
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
           
        self.pred = tf.nn.sigmoid(self.logits)
        self.pred_int = tf.cast(tf.greater(self.pred, 0.5), tf.float32)
        
        with tf.name_scope('stream_var'):
            self.acc, self.ac_update = tf.metrics.accuracy(self.y, self.pred_int)
            self.p, self.p_update = tf.metrics.precision(self.y, self.pred_int)
            self.r, self.r_update = tf.metrics.recall(self.y, self.pred_int)
            """
            self.p = [0] * self.class_n
            self.r = [0] * self.class_n
            self.p_update = [[]] * self.class_n
            self.r_update = [[]] * self.class_n
            for i in range(self.class_n):
            self.p[i], self.p_update[i] = tf.metrics.precision(tf.equal(self.y, i), tf.equal(self.pred, i))
            self.r[i], self.r_update[i] = tf.metrics.recall(tf.equal(self.y, i), tf.equal(self.pred, i))
            """

        stream_vars = [i for i in tf.local_variables() if i.name.split('/')[0]=='stream_var']
        self.reset_op = tf.variables_initializer(stream_vars)

    def check(self, x, y):
        feed_dict = {self.x : x, self.y : y}
        output = [self.y, self.y_onehot, self.logits, self.pred]
        a, b, c, d = self.sess.run(output, feed_dict)
        print a
        print b
        print c
        print d

    def train(self):

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if self.model_type != 'DNN':
            self.sess.run(self.embd_init, {self.embedding_pretrain : self.utils.load_word_embedding()})
        step = 1
        loss = 0.0

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("\nload model from {} ...".format(ckpt.model_checkpoint_path))
            step = int(ckpt.model_checkpoint_path.split('-')[-1])

        print('\nStart training ...\n')

        for x, y, l in self.utils.get_train_batch():
            feed_dict = {self.x : x, self.y : y}
            if self.model_type == 'RNN':
               feed_dict[self.seq_len] = l
            output = [self.op, self.p_update, self.r_update, self.ac_update, self.loss]
            _, _, _, _, loss_temp = self.sess.run(output, feed_dict)

            loss += loss_temp

            #self.check(x, y)

            if step % self.print_step == 0:
                loss /= self.print_step
                p = self.sess.run(self.p)
                r = self.sess.run(self.r)
                acc = self.sess.run(self.acc)
                self.utils.print_val(step, acc, p, r, loss)

                self.sess.run(self.reset_op)
                loss = 0.0

            if step % self.save_step == 0:
                print("\nSaving model ...")
                self.saver.save(self.sess, self.model_path, global_step=step)

            if step >= self.max_step:
                if step % self.save_step != 0:
                    print("\nSaving model ...")
                    self.saver.save(self.sess, self.model_path, global_step=step)
                break

            step += 1

    def test(self, mode):
        if self.load != '':
            self.saver.restore(self.sess, self.load)
            print("load model from {} ...".format(self.load))
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))

        self.sess.run(tf.local_variables_initializer())

        pred = []
        y = []

        for id, x, y, l in self.utils.get_test_batch(mode=mode):
            feed_dict = {self.x : x, self.y : y}
            if self.model_type == 'RNN':
                feed_dict[self.seq_len] = l
            pred, _, _, _ = self.sess.run([self.pred, self.p_update, self.r_update, self.ac_update], feed_dict)
            for i, j in zip(id, pred):
                j = [round(k, 2) for k in j]
                j.insert(0, i)
                self.utils.write_prediction(j)
        if mode == 'val':
            p = self.sess.run(self.p)
            r = self.sess.run(self.r)
            acc = self.sess.run(self.acc)
            self.utils.print_val(0, acc, p, r, 0)

