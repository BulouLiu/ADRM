# -*- coding: utf-8 -*-
__author__ = 'derrick.dy'
import tensorflow as tf
import numpy as np
import time
# import val_test_model
import movie_val_test_model

import sys
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)

id2word = {}

with open('./movie-review-5class/movie-review-0/word2id.txt', 'r', encoding='gbk') as f:
    for line in f:
        w, id = line.strip().split(' ')
        id2word[int(id)] = w
id2word[0] = 'PAD'


class DataGenerator(Configurable):
    # params for data generator
    max_q_len = Int(5, help='max q len').tag(config=True)
    max_d_len = Int(500, help='max document len').tag(config=True)
    q_name = Unicode('q')
    d_name = Unicode('d')
    q_str_name = Unicode('q_str')
    q_lens_name = Unicode('q_lens')
    aux_d_name = Unicode('d_aux')
    prf_d_1_name = Unicode('prf_1')
    prf_d_2_name = Unicode('prf_2')
    prf_d_3_name = Unicode('prf_3')
    prf_1_lens_name = Unicode('prf_1_lens')
    prf_2_lens_name = Unicode('prf_2_lens')
    prf_3_lens_name = Unicode('prf_3_lens')
    vocabulary_size = Int(2000000).tag(config=True)

    def __init__(self, **kwargs):
        # init the data generator
        super(DataGenerator, self).__init__(**kwargs)
        print("generator's vocabulary size: ", self.vocabulary_size)
        self.label_prf = movie_val_test_model.get_prf('movie_prf.txt')

    def pairwise_reader(self, pair_stream, batch_size, with_idf=False):
        # generate the batch of x,y in training time
        l_q = []
        l_q_str = []
        l_d = []
        l_d_aux = []
        l_y = []
        l_q_lens = []
        l_prf_1 = []
        l_prf_2 = []
        l_prf_3 = []
        l_prf_1_lens = []
        l_prf_2_lens = []
        l_prf_3_lens = []


        for line in pair_stream:

            cols = line.strip().split('\t')
            y = float(1.0)

            prf_doc = self.label_prf[cols[0]].strip().split(' ')

            l_q_str.append(cols[0])
            q = np.array([int(t) for t in cols[0].split(',') if int(t) < self.vocabulary_size])
            t1 = np.array([int(t) for t in cols[1].split(',') if int(t) < self.vocabulary_size])
            t2 = np.array([int(t) for t in cols[2].split(',') if int(t) < self.vocabulary_size])

            prf1 = np.array([int(t) for t in prf_doc[0].split(',') if int(t) < self.vocabulary_size])
            prf2 = np.array([int(t) for t in prf_doc[1].split(',') if int(t) < self.vocabulary_size])
            prf3 = np.array([int(t) for t in prf_doc[2].split(',') if int(t) < self.vocabulary_size])

            # padding 补全
            v_q = np.zeros(self.max_q_len)
            v_d = np.zeros(self.max_d_len)
            v_d_aux = np.zeros(self.max_d_len)
            v_prf_1 = np.zeros(self.max_d_len)
            v_prf_2 = np.zeros(self.max_d_len)
            v_prf_3 = np.zeros(self.max_d_len)

            v_q[:min(q.shape[0], self.max_q_len)] = q[:min(q.shape[0], self.max_q_len)]
            v_d[:min(t1.shape[0], self.max_d_len)] = t1[:min(t1.shape[0], self.max_d_len)]
            v_d_aux[:min(t2.shape[0], self.max_d_len)] = t2[:min(t2.shape[0], self.max_d_len)]
            v_prf_1[:min(prf1.shape[0], self.max_d_len)] = prf1[:min(prf1.shape[0], self.max_d_len)]
            v_prf_2[:min(prf2.shape[0], self.max_d_len)] = prf2[:min(prf2.shape[0], self.max_d_len)]
            v_prf_3[:min(prf3.shape[0], self.max_d_len)] = prf3[:min(prf3.shape[0], self.max_d_len)]

            l_q.append(v_q)
            l_d.append(v_d)
            l_d_aux.append(v_d_aux)
            l_y.append(y)
            l_q_lens.append(len(q))
            l_prf_1.append(v_prf_1)
            l_prf_2.append(v_prf_2)
            l_prf_3.append(v_prf_3)
            l_prf_1_lens.append(len(prf1))
            l_prf_2_lens.append(len(prf2))
            l_prf_3_lens.append(len(prf3))

            if len(l_q) >= batch_size:
                Q = np.array(l_q, dtype=int, )
                D = np.array(l_d, dtype=int, )
                D_aux = np.array(l_d_aux, dtype=int, )
                Q_lens = np.array(l_q_lens, dtype=int, )
                Y = np.array(l_y, dtype=int, )
                PRF_1 = np.array(l_prf_1, dtype=int, )
                PRF_2 = np.array(l_prf_2, dtype=int, )
                PRF_3 = np.array(l_prf_3, dtype=int, )
                PRF_1_lens = np.array(l_prf_1_lens, dtype=int, )
                PRF_2_lens = np.array(l_prf_2_lens, dtype=int, )
                PRF_3_lens = np.array(l_prf_3_lens, dtype=int, )
                X = {self.q_name: Q, self.d_name: D, self.aux_d_name: D_aux, self.q_lens_name: Q_lens,
                     self.q_str_name: l_q_str, self.prf_d_1_name: PRF_1, self.prf_d_2_name: PRF_2, self.prf_d_3_name:
                         PRF_3, self.prf_1_lens_name: PRF_1_lens, self.prf_2_lens_name: PRF_2_lens,
                     self.prf_3_lens_name: PRF_3_lens}
                yield X, Y
                l_q, l_d, l_d_aux, l_y, l_q_lens, l_ids, l_q_str, l_prf_1, l_prf_2, l_prf_3, l_prf_1_lens, \
                l_prf_2_lens, l_prf_3_lens = [], [], [], [], [], [], [], [], [], [], [], [], []
        if l_q:
            Q = np.array(l_q, dtype=int, )
            D = np.array(l_d, dtype=int, )
            D_aux = np.array(l_d_aux, dtype=int, )
            Q_lens = np.array(l_q_lens, dtype=int, )
            Y = np.array(l_y, dtype=int, )
            PRF_1 = np.array(l_prf_1, dtype=int, )
            PRF_2 = np.array(l_prf_2, dtype=int, )
            PRF_3 = np.array(l_prf_3, dtype=int, )
            PRF_1_lens = np.array(l_prf_1_lens, dtype=int, )
            PRF_2_lens = np.array(l_prf_2_lens, dtype=int, )
            PRF_3_lens = np.array(l_prf_3_lens, dtype=int, )
            X = {self.q_name: Q, self.d_name: D, self.aux_d_name: D_aux, self.q_lens_name: Q_lens,
                 self.q_str_name: l_q_str, self.prf_d_1_name: PRF_1, self.prf_d_2_name: PRF_2, self.prf_d_3_name:
                     PRF_3, self.prf_1_lens_name: PRF_1_lens, self.prf_2_lens_name: PRF_2_lens,
                 self.prf_3_lens_name: PRF_3_lens}
            yield X, Y

    def test_pairwise_reader(self, pair_stream, batch_size):
        # generate the batch of x,y in test time
        l_q = []
        l_q_lens = []
        l_d = []
        l_prf_1 = []
        l_prf_2 = []
        l_prf_3 = []
        l_prf_1_lens = []
        l_prf_2_lens = []
        l_prf_3_lens = []

        for line in pair_stream:
            cols = line.strip().split('\t')

            prf_doc = self.label_prf[cols[0]].strip().split(' ')

            q = np.array([int(t) for t in cols[0].split(',') if int(t) < self.vocabulary_size])
            t = np.array([int(t) for t in cols[1].split(',') if int(t) < self.vocabulary_size])

            prf1 = np.array([int(t) for t in prf_doc[0].split(',') if int(t) < self.vocabulary_size])
            prf2 = np.array([int(t) for t in prf_doc[1].split(',') if int(t) < self.vocabulary_size])
            prf3 = np.array([int(t) for t in prf_doc[2].split(',') if int(t) < self.vocabulary_size])

            v_q = np.zeros(self.max_q_len)
            v_d = np.zeros(self.max_d_len)
            v_prf_1 = np.zeros(self.max_d_len)
            v_prf_2 = np.zeros(self.max_d_len)
            v_prf_3 = np.zeros(self.max_d_len)

            v_q[:min(q.shape[0], self.max_q_len)] = q[:min(q.shape[0], self.max_q_len)]
            v_d[:min(t.shape[0], self.max_d_len)] = t[:min(t.shape[0], self.max_d_len)]
            v_prf_1[:min(prf1.shape[0], self.max_d_len)] = prf1[:min(prf1.shape[0], self.max_d_len)]
            v_prf_2[:min(prf2.shape[0], self.max_d_len)] = prf2[:min(prf2.shape[0], self.max_d_len)]
            v_prf_3[:min(prf3.shape[0], self.max_d_len)] = prf3[:min(prf3.shape[0], self.max_d_len)]

            l_q.append(v_q)
            l_d.append(v_d)
            l_q_lens.append(len(q))
            l_prf_1.append(v_prf_1)
            l_prf_2.append(v_prf_2)
            l_prf_3.append(v_prf_3)
            l_prf_1_lens.append(len(prf1))
            l_prf_2_lens.append(len(prf2))
            l_prf_3_lens.append(len(prf3))

            if len(l_q) >= batch_size:
                Q = np.array(l_q, dtype=int, )
                D = np.array(l_d, dtype=int, )
                Q_lens = np.array(l_q_lens, dtype=int, )
                PRF_1 = np.array(l_prf_1, dtype=int, )
                PRF_2 = np.array(l_prf_2, dtype=int, )
                PRF_3 = np.array(l_prf_3, dtype=int, )
                PRF_1_lens = np.array(l_prf_1_lens, dtype=int, )
                PRF_2_lens = np.array(l_prf_2_lens, dtype=int, )
                PRF_3_lens = np.array(l_prf_3_lens, dtype=int, )
                X = {self.q_name: Q, self.d_name: D, self.q_lens_name: Q_lens, self.prf_d_1_name: PRF_1,
                     self.prf_d_2_name: PRF_2, self.prf_d_3_name: PRF_3, self.prf_1_lens_name: PRF_1_lens,
                     self.prf_2_lens_name: PRF_2_lens, self.prf_3_lens_name: PRF_3_lens}
                yield X
                l_q, l_d, l_q_lens, l_prf_1, l_prf_2, l_prf_3, l_prf_1_lens, l_prf_2_lens, l_prf_3_lens \
                    = [], [], [], [], [], [], [], [], []
        if l_q:
            Q = np.array(l_q, dtype=int, )
            D = np.array(l_d, dtype=int, )
            Q_lens = np.array(l_q_lens, dtype=int, )
            PRF_1 = np.array(l_prf_1, dtype=int, )
            PRF_2 = np.array(l_prf_2, dtype=int, )
            PRF_3 = np.array(l_prf_3, dtype=int, )
            PRF_1_lens = np.array(l_prf_1_lens, dtype=int, )
            PRF_2_lens = np.array(l_prf_2_lens, dtype=int, )
            PRF_3_lens = np.array(l_prf_3_lens, dtype=int, )
            X = {self.q_name: Q, self.d_name: D, self.q_lens_name: Q_lens, self.prf_d_1_name: PRF_1,
                 self.prf_d_2_name: PRF_2, self.prf_d_3_name: PRF_3,
                 self.prf_1_lens_name: PRF_1_lens, self.prf_2_lens_name: PRF_2_lens, self.prf_3_lens_name: PRF_3_lens}
            yield X


class BaseNN(Configurable):
    # params of base deeprank model
    max_q_len = Int(5, help='max q len')
    max_d_len = Int(500, help='max document len')
    batch_size = Int(16, help="minibatch size")
    batch_test_size = Int(1, help="minibatch size")
    max_epochs = Float(50, help="maximum number of epochs")
    eval_frequency = Int(5, help="print out minibatch every * steps")
    checkpoint_steps = Int(5, help="stroe trained data every * steps")
    embedding_size = Int(300, help="embedding dimension")
    vocabulary_size = Int(55449, help="vocabulary size")
    kernal_width = Int(5, help='kernal width')
    kernal_num = Int(50, help='number of kernal')
    regular_term = Float(0.0001, help='param for controlling wight of L2 loss')
    maxpooling_num = Int(3, help='number of maxpooling')
    decoder_mlp1_num = Int(75, help='number of decoder mlp 1')
    decoder_mlp2_num = Int(1, help='number of decoder mlp 2')

    def __init__(self, **kwargs):
        super(BaseNN, self).__init__(**kwargs)
        # generator
        self.data_generator = DataGenerator()
        # validation in training stage is full test data in 20ng
        self.val_data_generator = DataGenerator()
        # test is zeros shot test data in 20ng (delete docs of zero shot label)
        self.test_data_generator = DataGenerator()

    @staticmethod
    def weight_variable(shape, name):
        tmp = np.sqrt(3.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random_uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial_value=initial, name=name)

    @staticmethod
    def re_pad(D, batch_size):
        D = np.array(D)
        D[D < 0] = 0
        if len(D) < batch_size:
            tmp = np.zeros((batch_size - len(D), D.shape[1]))
            D = np.concatenate((D, tmp), axis=0)
        return D

    def gen_query_mask(self, Q):
        mask = np.zeros((self.batch_size, self.max_q_len))
        for b in range(len(Q)):
            for q in range(len(Q[b])):
                if Q[b][q] > 0:
                    mask[b][q] = 1

        return mask

    def gen_test_query_mask(self, Q):
        mask = np.zeros((self.batch_test_size, self.max_q_len))
        for b in range(len(Q)):
            for q in range(len(Q[b])):
                if Q[b][q] > 0:
                    mask[b][q] = 1

        return mask

    def gen_doc_mask(self, D):
        mask = np.zeros((self.batch_size, self.max_d_len))
        for b in range(len(D)):
            for q in range(len(D[b])):
                if D[b][q] > 0:
                    mask[b][q] = 1

        return mask

    def gen_test_doc_mask(self, D):
        mask = np.zeros((self.batch_test_size, self.max_d_len))
        for b in range(len(D)):
            for q in range(len(D[b])):
                if D[b][q] > 0:
                    mask[b][q] = 1

        return mask


class ZSL_TC(BaseNN):
    # params of zeroshot-text classification model
    neg_sample = 1  # num of neg sample when training

    emb_in = Unicode('./movie-review-5class/movie-review-0/review-glove.txt',
                     help="initial embedding. Terms should be hashed to ids.")
    model_learning_rate = Float(0.00001, help="learning rate of model, default is 0.001")
    adv_learning_rate = Float(0.00001, help='learning rate of adv classifier, default is 0.001')
    epsilon = Float(0.00001, help="Epsilon for Adam")
    val_realscore_path = Unicode('None', help='val realscore file path')
    test_realscore_path = Unicode('./movie-review-5class/movie-review-0/movie-test-realscore.txt',
                                  help='test realscore file path')
    label_dict_path = Unicode('./movie-review-5class/movie-review-0/movie-label-dict.txt', help='label dict path')
    word2id_path = Unicode('./movie-review-5class/movie-review-0/word2id.txt', help='word2id path')
    # test_type = Int(-1, help='test type').tag(config=True)
    train_class_num = Int(3, help='num of train class')
    adv_term = Float(0.1, help='regular term of adversrial loss')
    zsl_num = Int(1)
    zsl_type = Int(0)

    def __init__(self, **kwargs):
        super(ZSL_TC, self).__init__(**kwargs)
        print("trying to load initial embeddings from:  ", self.emb_in)
        if self.emb_in != 'None':
            self.emb = self.load_word2vec(self.emb_in)
            self.embeddings = tf.Variable(
                tf.constant(self.emb, dtype='float32', shape=[self.vocabulary_size + 1, self.embedding_size]),
                trainable=False)
            print("Initialized embeddings with {0}".format(self.emb_in))
        else:
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size + 1, self.embedding_size], -1.0, 1.0))


        self.query_gate_weight = BaseNN.weight_variable((self.embedding_size, self.kernal_num), 'gate_weight')
        self.query_gate_bias = tf.Variable(initial_value=tf.zeros((self.kernal_num)), name='gate_bias')
        self.adv_weight = BaseNN.weight_variable((self.decoder_mlp1_num, self.train_class_num), name='adv_weight')
        self.adv_bias = tf.Variable(initial_value=tf.zeros((1, self.train_class_num)), name='adv_bias')
        # get the label information to help adversarial learning

        self.label_dict, self.reverse_label_dict, self.label_list = movie_val_test_model.get_labels(self.label_dict_path,
                                                                                         self.word2id_path)
        self.label_index_dict = movie_val_test_model.get_label_index(self.label_list, self.zsl_type)

    def load_word2vec(self, emb_file_path):
        emb = np.zeros((self.vocabulary_size + 1, self.embedding_size))
        nlines = 0
        with open(emb_file_path) as f:
            for line in f:
                nlines += 1
                if nlines == 1:
                    continue
                items = line.split()
                tid = int(items[0])
                if tid > self.vocabulary_size:
                    print(tid)
                    continue
                vec = np.array([float(t) for t in items[1:]])
                emb[tid, :] = vec
                if nlines % 20000 == 0:
                    print("load {0} vectors...".format(nlines))

        return emb

    def gen_adv_query_mask(self, q_ids):
        q_mask = np.zeros((self.batch_size, self.train_class_num))
        for batch_num, b_q_id in enumerate(q_ids):
            c_name = self.reverse_label_dict[b_q_id]
            c_index = self.label_index_dict[c_name]
            q_mask[batch_num][c_index] = 1
        return q_mask

    def get_class_gate(self, class_vec, emb_d):
        '''
        compute the gate in kernal space
        :param class_vec: avg emb of seed words
        :param emb_d: emb of doc
        :return:the class gate [batchsize,d_len,kernal_num]
        '''
        gate1 = tf.expand_dims(tf.matmul(class_vec, self.query_gate_weight), axis=1)
        bias = tf.expand_dims(self.query_gate_bias, axis=0)
        gate = tf.add(gate1, bias)
        return tf.sigmoid(gate)

    def L2_model_loss(self):
        all_para = [v for v in tf.trainable_variables() if 'b' not in v.name and 'adv' not in v.name]
        loss = 0.
        for each in all_para:
            loss += tf.nn.l2_loss(each)
        return loss

    def L2_adv_loss(self):
        all_para = [v for v in tf.trainable_variables() if 'b' not in v.name and 'adv' in v.name]
        loss = 0.
        for each in all_para:
            loss += tf.nn.l2_loss(each)
        return loss

    def softmax(self, target, axis, mask, epsilon=1e-12, name=None):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis) * mask
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / (normalize + epsilon)
        return softmax

    def train(self, train_pair_file_path, val_pair_file_path, checkpoint_dir, load_model=False):

        input_q = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len])
        input_pos_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len])
        input_neg_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len])
        input_prf_1 = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len])
        input_prf_2 = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len])
        input_prf_3 = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len])
        q_lens = tf.placeholder(tf.float32, shape=[self.batch_size, ])
        prf_1_lens = tf.placeholder(tf.float32, shape=[self.batch_size, ])
        prf_2_lens = tf.placeholder(tf.float32, shape=[self.batch_size, ])
        prf_3_lens = tf.placeholder(tf.float32, shape=[self.batch_size, ])
        q_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len])
        pos_d_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len])
        neg_d_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len])
        input_q_index = tf.placeholder(tf.int32, shape=[self.batch_size, self.train_class_num])
        prf_1_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len])
        prf_2_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len])
        prf_3_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len])
        emb_q = tf.nn.embedding_lookup(self.embeddings, input_q)
        emb_prf_1 = tf.nn.embedding_lookup(self.embeddings, input_prf_1)
        emb_prf_2 = tf.nn.embedding_lookup(self.embeddings, input_prf_2)
        emb_prf_3 = tf.nn.embedding_lookup(self.embeddings, input_prf_3)

        emb_q_attention = tf.transpose(emb_q, perm=[0, 2, 1])

        similarity_m1 = tf.matmul(emb_prf_1, emb_q_attention)
        similarity_m2 = tf.matmul(emb_prf_2, emb_q_attention)
        similarity_m3 = tf.matmul(emb_prf_3, emb_q_attention)

        M_mask1 = tf.to_float(tf.matmul(tf.expand_dims(prf_1_mask, -1), tf.expand_dims(q_mask, 1)))
        M_mask2 = tf.to_float(tf.matmul(tf.expand_dims(prf_2_mask, -1), tf.expand_dims(q_mask, 1)))
        M_mask3 = tf.to_float(tf.matmul(tf.expand_dims(prf_3_mask, -1), tf.expand_dims(q_mask, 1)))

        alpha1 = self.softmax(similarity_m1, 1, M_mask1)
        alpha2 = self.softmax(similarity_m2, 1, M_mask2)
        alpha3 = self.softmax(similarity_m3, 1, M_mask3)

        prf1_importance = tf.expand_dims(tf.reduce_max(alpha1, 2), -1)
        prf2_importance = tf.expand_dims(tf.reduce_max(alpha2, 2), -1)
        prf3_importance = tf.expand_dims(tf.reduce_max(alpha3, 2), -1)

        prf_1_vec_sum = tf.reduce_sum(
            tf.multiply(emb_prf_1, prf1_importance),
            axis=1
        )
        prf_2_vec_sum = tf.reduce_sum(
            tf.multiply(emb_prf_2, prf2_importance),
            axis=1
        )
        prf_3_vec_sum = tf.reduce_sum(
            tf.multiply(emb_prf_3, prf3_importance),
            axis=1
        )

        class_vec_sum = tf.reduce_sum(
            tf.multiply(emb_q, tf.expand_dims(q_mask, axis=-1)),
            axis=1
        )
        
        class_vec = tf.div(class_vec_sum, tf.expand_dims(q_lens, -1))
        prf_1_vec = tf.div(prf_1_vec_sum, tf.expand_dims(prf_1_lens, -1))
        prf_2_vec = tf.div(prf_2_vec_sum, tf.expand_dims(prf_2_lens, -1))
        prf_3_vec = tf.div(prf_3_vec_sum, tf.expand_dims(prf_3_lens, -1))

        emb_pos_d = tf.nn.embedding_lookup(self.embeddings, input_pos_d)
        emb_neg_d = tf.nn.embedding_lookup(self.embeddings, input_neg_d)

        pos_query_gate = self.get_class_gate(class_vec, emb_pos_d)
        neg_query_gate = self.get_class_gate(class_vec, emb_neg_d)
        pos_prf_1_gate = self.get_class_gate(prf_1_vec, emb_pos_d)
        neg_prf_1_gate = self.get_class_gate(prf_1_vec, emb_neg_d)
        pos_prf_2_gate = self.get_class_gate(prf_2_vec, emb_pos_d)
        neg_prf_2_gate = self.get_class_gate(prf_2_vec, emb_neg_d)
        pos_prf_3_gate = self.get_class_gate(prf_3_vec, emb_pos_d)
        neg_prf_3_gate = self.get_class_gate(prf_3_vec, emb_neg_d)

        # CNN for document
        pos_mult_info = tf.multiply(tf.expand_dims(class_vec, axis=1), emb_pos_d)
        pos_sub_info = tf.expand_dims(class_vec, axis=1) - emb_pos_d
        pos_conv_input = tf.concat([emb_pos_d, pos_mult_info, pos_sub_info], axis=-1)
        neg_mult_info = tf.multiply(tf.expand_dims(class_vec, axis=1), emb_neg_d)
        neg_sub_info = tf.expand_dims(class_vec, axis=1) - emb_neg_d
        neg_conv_input = tf.concat([emb_neg_d, neg_mult_info, neg_sub_info], axis=-1)

        pos_prf_1_mult_info = tf.multiply(tf.expand_dims(prf_1_vec, axis=1), emb_pos_d)
        pos_prf_1_sub_info = tf.expand_dims(prf_1_vec, axis=1) - emb_pos_d
        pos_prf_1_conv_input = tf.concat([emb_pos_d, pos_prf_1_mult_info, pos_prf_1_sub_info], axis=-1)
        neg_prf_1_mult_info = tf.multiply(tf.expand_dims(prf_1_vec, axis=1), emb_neg_d)
        neg_prf_1_sub_info = tf.expand_dims(prf_1_vec, axis=1) - emb_neg_d
        neg_prf_1_conv_input = tf.concat([emb_neg_d, neg_prf_1_mult_info, neg_prf_1_sub_info], axis=-1)

        pos_prf_2_mult_info = tf.multiply(tf.expand_dims(prf_2_vec, axis=1), emb_pos_d)
        pos_prf_2_sub_info = tf.expand_dims(prf_2_vec, axis=1) - emb_pos_d
        pos_prf_2_conv_input = tf.concat([emb_pos_d, pos_prf_2_mult_info, pos_prf_2_sub_info], axis=-1)
        neg_prf_2_mult_info = tf.multiply(tf.expand_dims(prf_2_vec, axis=1), emb_neg_d)
        neg_prf_2_sub_info = tf.expand_dims(prf_2_vec, axis=1) - emb_neg_d
        neg_prf_2_conv_input = tf.concat([emb_neg_d, neg_prf_2_mult_info, neg_prf_2_sub_info], axis=-1)

        pos_prf_3_mult_info = tf.multiply(tf.expand_dims(prf_3_vec, axis=1), emb_pos_d)
        pos_prf_3_sub_info = tf.expand_dims(prf_3_vec, axis=1) - emb_pos_d
        pos_prf_3_conv_input = tf.concat([emb_pos_d, pos_prf_3_mult_info, pos_prf_3_sub_info], axis=-1)
        neg_prf_3_mult_info = tf.multiply(tf.expand_dims(prf_3_vec, axis=1), emb_neg_d)
        neg_prf_3_sub_info = tf.expand_dims(prf_3_vec, axis=1) - emb_neg_d
        neg_prf_3_conv_input = tf.concat([emb_neg_d, neg_prf_3_mult_info, neg_prf_3_sub_info], axis=-1)

        # shape: (1, 500, 900)

        # in fact that's 1D conv, but we implement it by conv2d
        pos_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(pos_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_conv'
        )
        neg_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(neg_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_conv',
            reuse=True
        )
        pos_prf_1_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(pos_prf_1_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],  
            padding='SAME',
            trainable=True,
            name='doc_prf_1_conv'
        )
        neg_prf_1_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(neg_prf_1_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_prf_1_conv',
            reuse=True
        )
        pos_prf_2_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(pos_prf_2_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],  
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_prf_2_conv'
        )
        neg_prf_2_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(neg_prf_2_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_prf_2_conv',
            reuse=True
        )
        pos_prf_3_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(pos_prf_3_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],  
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_prf_3_conv'
        )
        neg_prf_3_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(neg_prf_3_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_prf_3_conv',
            reuse=True
        )
        # shape=[batch,max_dlen,1,kernal_num]
        # reshape to [batch,max_dlen,kernal_num]


        rs_pos_conv = tf.squeeze(pos_conv)
        rs_neg_conv = tf.squeeze(neg_conv)
        rs_pos_prf_1_conv = tf.squeeze(pos_prf_1_conv)
        rs_neg_prf_1_conv = tf.squeeze(neg_prf_1_conv)
        rs_pos_prf_2_conv = tf.squeeze(pos_prf_2_conv)
        rs_neg_prf_2_conv = tf.squeeze(neg_prf_2_conv)
        rs_pos_prf_3_conv = tf.squeeze(pos_prf_3_conv)
        rs_neg_prf_3_conv = tf.squeeze(neg_prf_3_conv)

        # query_gate elment-wise multiply rs_pos_conv
        pos_gate_conv = tf.multiply(pos_query_gate, rs_pos_conv)
        neg_gate_conv = tf.multiply(neg_query_gate, rs_neg_conv)
        pos_prf_1_gate_conv = tf.multiply(pos_prf_1_gate, rs_pos_prf_1_conv)
        neg_prf_1_gate_conv = tf.multiply(neg_prf_1_gate, rs_neg_prf_1_conv)
        pos_prf_2_gate_conv = tf.multiply(pos_prf_2_gate, rs_pos_prf_2_conv)
        neg_prf_2_gate_conv = tf.multiply(neg_prf_2_gate, rs_neg_prf_2_conv)
        pos_prf_3_gate_conv = tf.multiply(pos_prf_3_gate, rs_pos_prf_3_conv)
        neg_prf_3_gate_conv = tf.multiply(neg_prf_3_gate, rs_neg_prf_3_conv)

        # K-max_pooling
        # transpose to [batch,knum,dlen],then get max k in each kernal filter
        transpose_pos_gate_conv = tf.transpose(pos_gate_conv, perm=[0, 2, 1])
        transpose_neg_gate_conv = tf.transpose(neg_gate_conv, perm=[0, 2, 1])
        transpose_pos_prf_1_gate_conv = tf.transpose(pos_prf_1_gate_conv, perm=[0, 2, 1])
        transpose_neg_prf_1_gate_conv = tf.transpose(neg_prf_1_gate_conv, perm=[0, 2, 1])
        transpose_pos_prf_2_gate_conv = tf.transpose(pos_prf_2_gate_conv, perm=[0, 2, 1])
        transpose_neg_prf_2_gate_conv = tf.transpose(neg_prf_2_gate_conv, perm=[0, 2, 1])
        transpose_pos_prf_3_gate_conv = tf.transpose(pos_prf_3_gate_conv, perm=[0, 2, 1])
        transpose_neg_prf_3_gate_conv = tf.transpose(neg_prf_3_gate_conv, perm=[0, 2, 1])

        # shape = [batch,k_num,maxpolling_num]
        # the k-max pooling here is implemented by function top_k, so the relative position information is ignored
        pos_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_pos_gate_conv,
            k=self.maxpooling_num,
        )
        neg_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_neg_gate_conv,
            k=self.maxpooling_num,
        )
        pos_prf_1_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_pos_prf_1_gate_conv,
            k=self.maxpooling_num,
        )
        neg_prf_1_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_neg_prf_1_gate_conv,
            k=self.maxpooling_num,
        )
        pos_prf_2_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_pos_prf_2_gate_conv,
            k=self.maxpooling_num,
        )
        neg_prf_2_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_neg_prf_2_gate_conv,
            k=self.maxpooling_num,
        )
        pos_prf_3_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_pos_prf_3_gate_conv,
            k=self.maxpooling_num,
        )
        neg_prf_3_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_neg_prf_3_gate_conv,
            k=self.maxpooling_num,
        )


        pos_encoder = tf.reshape(pos_kmaxpooling, shape=(self.batch_size, -1))
        neg_encoder = tf.reshape(neg_kmaxpooling, shape=(self.batch_size, -1))
        pos_prf_1_encoder = tf.reshape(pos_prf_1_kmaxpooling, shape=(self.batch_size, -1))
        neg_prf_1_encoder = tf.reshape(neg_prf_1_kmaxpooling, shape=(self.batch_size, -1))
        pos_prf_2_encoder = tf.reshape(pos_prf_2_kmaxpooling, shape=(self.batch_size, -1))
        neg_prf_2_encoder = tf.reshape(neg_prf_2_kmaxpooling, shape=(self.batch_size, -1))
        pos_prf_3_encoder = tf.reshape(pos_prf_3_kmaxpooling, shape=(self.batch_size, -1))
        neg_prf_3_encoder = tf.reshape(neg_prf_3_kmaxpooling, shape=(self.batch_size, -1))

        pos_all_encoder = tf.concat([pos_encoder, pos_prf_1_encoder, pos_prf_2_encoder, pos_prf_3_encoder], axis=1)
        neg_all_encoder = tf.concat([neg_encoder, neg_prf_1_encoder, neg_prf_2_encoder, neg_prf_3_encoder], axis=1)

        pos_decoder_mlp1 = tf.layers.dense(
            inputs=pos_all_encoder,
            units=self.decoder_mlp1_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp1'
        )

        neg_decoder_mlp1 = tf.layers.dense(
            inputs=neg_all_encoder,
            units=self.decoder_mlp1_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp1',
            reuse=True
        )

        pos_decoder_mlp2 = tf.layers.dense(
            inputs=pos_decoder_mlp1,
            units=self.decoder_mlp2_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp2'
        )

        neg_decoder_mlp2 = tf.layers.dense(
            inputs=neg_decoder_mlp1,
            units=self.decoder_mlp2_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp2',
            reuse=True
        )

        score_pos = pos_decoder_mlp2
        score_neg = neg_decoder_mlp2

        hinge_loss = tf.reduce_mean(tf.maximum(0.0, 1 - score_pos + score_neg))

        adv_prob = tf.nn.softmax(tf.add(tf.matmul(pos_decoder_mlp1, self.adv_weight), self.adv_bias))
        log_adv_prob = tf.log(adv_prob)
        adv_loss = tf.reduce_mean(
            tf.reduce_sum(tf.multiply(log_adv_prob, tf.cast(input_q_index, tf.float32)), axis=1, keep_dims=True))
        L2_adv_loss = self.regular_term * self.L2_adv_loss()

        # to apply GRL, we use two seperate optimizers for adversarial classifier and the rest part of DAZER
        # optimizer for adversarial classifier
        adv_var_list = [v for v in tf.trainable_variables() if 'adv' in v.name]
        adv_opt = tf.train.AdamOptimizer(learning_rate=self.adv_learning_rate, epsilon=self.epsilon).minimize(
            loss=(-1 * adv_loss + L2_adv_loss), var_list=adv_var_list)

        # optimizer for rest part of DAZER model
        L2_model_loss = self.regular_term * self.L2_model_loss()
        model_var_list = [v for v in tf.trainable_variables() if 'adv' not in v.name]
        loss = hinge_loss + L2_model_loss + (adv_loss * self.adv_term)
        model_opt = tf.train.AdamOptimizer(learning_rate=self.model_learning_rate, epsilon=self.epsilon).minimize(
            loss=loss, var_list=model_var_list)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        val_results = []
        save_num = 0
        save_var = [v for v in tf.trainable_variables()]

        # Create a local session to run the training.
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(max_to_keep=50, var_list=save_var)
            start_time = time.time()
            if not load_model:
                print("Initializing a new model...")
                init = tf.global_variables_initializer()
                sess.run(init)
                print('New model initialized!')
            else:
                # to load trained model, and keep training
                # remember to change the name of ckpt file
                init = tf.global_variables_initializer()
                sess.run(init)
                saver.restore(sess, checkpoint_dir + '/zsl25.ckpt')
                print("model loaded!")

            # Loop through training steps.
            step = 0
            loss_list = []

            for epoch in range(int(self.max_epochs)):
                epoch_val_loss = 0
                epoch_loss = 0
                epoch_hinge_loss = 0.
                epoch_adv_loss = 0
                epoch_s = time.time()
                pair_stream = open(train_pair_file_path)

                for BATCH in self.data_generator.pairwise_reader(pair_stream, self.batch_size):
                    step += 1
                    X, Y = BATCH
                    query = X[u'q']
                    str_query = X[u'q_str']
                    q_index = self.gen_adv_query_mask(str_query)
                    pos_doc = X[u'd']
                    neg_doc = X[u'd_aux']
                    prf_1 = X[u'prf_1']
                    prf_2 = X[u'prf_2']
                    prf_3 = X[u'prf_3']
                    train_q_lens = X[u'q_lens']
                    train_prf_1_lens = X[u'prf_1_lens']
                    train_prf_2_lens = X[u'prf_2_lens']
                    train_prf_3_lens = X[u'prf_3_lens']
                    M_query = self.gen_query_mask(query)
                    M_pos = self.gen_doc_mask(pos_doc)
                    M_neg = self.gen_doc_mask(neg_doc)
                    M_prf_1 = self.gen_doc_mask(prf_1)
                    M_prf_2 = self.gen_doc_mask(prf_2)
                    M_prf_3 = self.gen_doc_mask(prf_3)

                    if X[u'q_lens'].shape[0] != self.batch_size:
                        continue
                    train_feed_dict = {input_q: query,
                                       input_pos_d: pos_doc,
                                       q_lens: train_q_lens,
                                       input_neg_d: neg_doc,
                                       q_mask: M_query,
                                       pos_d_mask: M_pos,
                                       neg_d_mask: M_neg,
                                       input_q_index: q_index,
                                       input_prf_1: prf_1,
                                       input_prf_2: prf_2,
                                       input_prf_3: prf_3,
                                       prf_1_lens: train_prf_1_lens,
                                       prf_2_lens: train_prf_2_lens,
                                       prf_3_lens: train_prf_3_lens,
                                       prf_1_mask: M_prf_1,
                                       prf_2_mask: M_prf_2,
                                       prf_3_mask: M_prf_3
                                       }

                    _1, l, hinge_l, _2, adv_l = sess.run([model_opt, loss, hinge_loss, adv_opt, adv_loss],
                                                         feed_dict=train_feed_dict)
                    epoch_loss += l
                    epoch_hinge_loss += hinge_l
                    epoch_adv_loss += adv_l

                if (epoch + 1) % self.eval_frequency == 0:
                    # after eval_frequency epochs we run model on val dataset
                    val_start = time.time()
                    val_pair_stream = open(val_pair_file_path)
                    for BATCH in self.val_data_generator.pairwise_reader(val_pair_stream, self.batch_size):
                        X_val, Y_val = BATCH
                        query = X_val[u'q']
                        pos_doc = X_val[u'd']
                        neg_doc = X_val[u'd_aux']
                        prf_1 = X_val[u'prf_1']
                        prf_2 = X_val[u'prf_2']
                        prf_3 = X_val[u'prf_3']
                        val_q_lens = X_val[u'q_lens']
                        val_prf_1_lens = X_val[u'prf_1_lens']
                        val_prf_2_lens = X_val[u'prf_2_lens']
                        val_prf_3_lens = X_val[u'prf_3_lens']
                        M_query = self.gen_query_mask(query)
                        M_pos = self.gen_doc_mask(pos_doc)
                        M_neg = self.gen_doc_mask(neg_doc)
                        M_prf_1 = self.gen_doc_mask(prf_1)
                        M_prf_2 = self.gen_doc_mask(prf_2)
                        M_prf_3 = self.gen_doc_mask(prf_3)
                        if X_val[u'q'].shape[0] != self.batch_size:
                            continue
                        train_feed_dict = {input_q: query,
                                           input_pos_d: pos_doc,
                                           input_neg_d: neg_doc,
                                           q_lens: val_q_lens,
                                           q_mask: M_query,
                                           pos_d_mask: M_pos,
                                           neg_d_mask: M_neg,
                                           input_prf_1: prf_1,
                                           input_prf_2: prf_2,
                                           input_prf_3: prf_3,
                                           prf_1_lens: val_prf_1_lens,
                                           prf_2_lens: val_prf_2_lens,
                                           prf_3_lens: val_prf_3_lens,
                                           prf_1_mask: M_prf_1,
                                           prf_2_mask: M_prf_2,
                                           prf_3_mask: M_prf_3
                                           }
                        #
                        # print(input_prf_1.shape, prf_1.shape)
                        # print(input_prf_2.shape, prf_2.shape)
                        # print(input_prf_3.shape, prf_3.shape)
                        # print(prf_1_lens.shape, val_prf_1_lens.shape)
                        # print(prf_2_lens.shape, val_prf_2_lens.shape)
                        # print(prf_3_lens.shape, val_prf_3_lens.shape)

                        # Run the graph and fetch some of the nodes.
                        v_loss = sess.run(hinge_loss, feed_dict=train_feed_dict)
                        epoch_val_loss += v_loss
                        val_results.append(epoch_val_loss)

                    val_end = time.time()
                    print('---Validation:epoch %d, %.1f ms , val_loss are %f' % (
                    epoch + 1, val_end - val_start, epoch_val_loss))
                    sys.stdout.flush()
                loss_list.append(epoch_loss)
                epoch_e = time.time()
                print('---Train:%d epoches cost %f seconds, hinge cost = %f  model cost = %f, adv cost = %f...' % (
                epoch + 1, epoch_e - epoch_s, epoch_hinge_loss, epoch_loss, epoch_adv_loss))
                # save model after checkpoint_steps epochs
                if (epoch + 1) % self.checkpoint_steps == 0:
                    save_num += 1
                    saver.save(sess, checkpoint_dir + 'zsl' + str(epoch + 1) + '.ckpt')
                pair_stream.close()

            with open('save_training_loss.txt', 'w') as f:
                for index, _loss in enumerate(loss_list):
                    f.write('epoch' + str(index + 1) + ', loss:' + str(_loss) + '\n')

            with open('save_val_cost.txt', 'w') as f:
                for index, v_l in enumerate(val_results):
                    f.write('epoch' + str((index + 1) * self.eval_frequency) + ' val loss:' + str(v_l) + '\n')

            # end training
            end_time = time.time()
            print('All costs %f seconds...' % (end_time - start_time))

    def test(self, test_point_file_path, test_size, output_file_path, checkpoint_dir=None, load_model=False):

        input_q = tf.placeholder(tf.int32, shape=[self.batch_test_size, self.max_q_len])
        input_pos_d = tf.placeholder(tf.int32, shape=[self.batch_test_size, self.max_d_len])
        input_prf_1 = tf.placeholder(tf.int32, shape=[self.batch_test_size, self.max_d_len])
        input_prf_2 = tf.placeholder(tf.int32, shape=[self.batch_test_size, self.max_d_len])
        input_prf_3 = tf.placeholder(tf.int32, shape=[self.batch_test_size, self.max_d_len])
        q_lens = tf.placeholder(tf.float32, shape=[self.batch_test_size, ])
        prf_1_lens = tf.placeholder(tf.float32, shape=[self.batch_test_size, ])
        prf_2_lens = tf.placeholder(tf.float32, shape=[self.batch_test_size, ])
        prf_3_lens = tf.placeholder(tf.float32, shape=[self.batch_test_size, ])
        q_mask = tf.placeholder(tf.float32, shape=[self.batch_test_size, self.max_q_len])
        pos_d_mask = tf.placeholder(tf.float32, shape=[self.batch_test_size, self.max_d_len])
        prf_1_mask = tf.placeholder(tf.float32, shape=[self.batch_test_size, self.max_d_len])
        prf_2_mask = tf.placeholder(tf.float32, shape=[self.batch_test_size, self.max_d_len])
        prf_3_mask = tf.placeholder(tf.float32, shape=[self.batch_test_size, self.max_d_len])

        emb_q = tf.nn.embedding_lookup(self.embeddings, input_q)
        emb_prf_1 = tf.nn.embedding_lookup(self.embeddings, input_prf_1)
        emb_prf_2 = tf.nn.embedding_lookup(self.embeddings, input_prf_2)
        emb_prf_3 = tf.nn.embedding_lookup(self.embeddings, input_prf_3)

        emb_q_attention = tf.transpose(emb_q, perm=[0, 2, 1])

        # (16, 500, 5)
        similarity_m1 = tf.matmul(emb_prf_1, emb_q_attention)
        similarity_m2 = tf.matmul(emb_prf_2, emb_q_attention)
        similarity_m3 = tf.matmul(emb_prf_3, emb_q_attention)

        M_mask1 = tf.to_float(tf.matmul(tf.expand_dims(prf_1_mask, -1), tf.expand_dims(q_mask, 1)))
        M_mask2 = tf.to_float(tf.matmul(tf.expand_dims(prf_2_mask, -1), tf.expand_dims(q_mask, 1)))
        M_mask3 = tf.to_float(tf.matmul(tf.expand_dims(prf_3_mask, -1), tf.expand_dims(q_mask, 1)))

        alpha1 = self.softmax(similarity_m1, 1, M_mask1)
        alpha2 = self.softmax(similarity_m2, 1, M_mask2)
        alpha3 = self.softmax(similarity_m3, 1, M_mask3)

        prf1_importance = tf.expand_dims(tf.reduce_max(alpha1, 2), -1)
        prf2_importance = tf.expand_dims(tf.reduce_max(alpha2, 2), -1)
        prf3_importance = tf.expand_dims(tf.reduce_max(alpha3, 2), -1)

        prf_1_vec_sum = tf.reduce_sum(
            tf.multiply(emb_prf_1, prf1_importance),
            axis=1
        )
        prf_2_vec_sum = tf.reduce_sum(
            tf.multiply(emb_prf_2, prf2_importance),
            axis=1
        )
        prf_3_vec_sum = tf.reduce_sum(
            tf.multiply(emb_prf_3, prf3_importance),
            axis=1
        )

        class_vec_sum = tf.reduce_sum(
            tf.multiply(emb_q, tf.expand_dims(q_mask, axis=-1)),
            axis=1
        )


        class_vec = tf.div(class_vec_sum, tf.expand_dims(q_lens, axis=-1))
        prf_1_vec = tf.div(prf_1_vec_sum, tf.expand_dims(prf_1_lens, -1))
        prf_2_vec = tf.div(prf_2_vec_sum, tf.expand_dims(prf_2_lens, -1))
        prf_3_vec = tf.div(prf_3_vec_sum, tf.expand_dims(prf_3_lens, -1))

        emb_pos_d = tf.nn.embedding_lookup(self.embeddings, input_pos_d)

        # get query gate
        query_gate = self.get_class_gate(class_vec, emb_pos_d)
        pos_prf_1_gate = self.get_class_gate(prf_1_vec, emb_pos_d)
        pos_prf_2_gate = self.get_class_gate(prf_2_vec, emb_pos_d)
        pos_prf_3_gate = self.get_class_gate(prf_3_vec, emb_pos_d)

        pos_mult_info = tf.multiply(tf.expand_dims(class_vec, axis=1), emb_pos_d)
        pos_sub_info = tf.expand_dims(class_vec, axis=1) - emb_pos_d
        pos_conv_input = tf.concat([emb_pos_d, pos_mult_info, pos_sub_info], axis=-1)
        pos_prf_1_mult_info = tf.multiply(tf.expand_dims(prf_1_vec, axis=1), emb_pos_d)
        pos_prf_1_sub_info = tf.expand_dims(prf_1_vec, axis=1) - emb_pos_d
        pos_prf_1_conv_input = tf.concat([emb_pos_d, pos_prf_1_mult_info, pos_prf_1_sub_info], axis=-1)
        pos_prf_2_mult_info = tf.multiply(tf.expand_dims(prf_2_vec, axis=1), emb_pos_d)
        pos_prf_2_sub_info = tf.expand_dims(prf_2_vec, axis=1) - emb_pos_d
        pos_prf_2_conv_input = tf.concat([emb_pos_d, pos_prf_2_mult_info, pos_prf_2_sub_info], axis=-1)
        pos_prf_3_mult_info = tf.multiply(tf.expand_dims(prf_3_vec, axis=1), emb_pos_d)
        pos_prf_3_sub_info = tf.expand_dims(prf_3_vec, axis=1) - emb_pos_d
        pos_prf_3_conv_input = tf.concat([emb_pos_d, pos_prf_3_mult_info, pos_prf_3_sub_info], axis=-1)

        # CNN for document
        pos_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(pos_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3],
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_conv'
        )
        pos_prf_1_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(pos_prf_1_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3], 
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_prf_1_conv'
        )
        pos_prf_2_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(pos_prf_2_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3], 
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_prf_2_conv'
        )
        pos_prf_3_conv = tf.layers.conv2d(
            inputs=tf.expand_dims(pos_prf_3_conv_input, axis=-1),
            filters=self.kernal_num,
            kernel_size=[self.kernal_width, self.embedding_size * 3], 
            strides=[1, self.embedding_size * 3],
            padding='SAME',
            trainable=True,
            name='doc_prf_3_conv'
        )
        # shape=[batch,max_dlen,1,kernal_num]
        # reshape to [batch,max_dlen,kernal_num]
        rs_pos_conv = tf.squeeze(pos_conv)
        rs_pos_prf_1_conv = tf.squeeze(pos_prf_1_conv)
        rs_pos_prf_2_conv = tf.squeeze(pos_prf_2_conv)
        rs_pos_prf_3_conv = tf.squeeze(pos_prf_3_conv)

        # query_gate elment-wise multiply rs_pos_conv
        # [batch,kernal_num] , [batch,max_dlen,kernal_num]
        pos_gate_conv = tf.multiply(query_gate, rs_pos_conv)
        pos_prf_1_gate_conv = tf.multiply(pos_prf_1_gate, rs_pos_prf_1_conv)
        pos_prf_2_gate_conv = tf.multiply(pos_prf_2_gate, rs_pos_prf_2_conv)
        pos_prf_3_gate_conv = tf.multiply(pos_prf_3_gate, rs_pos_prf_3_conv)

        # K-max_pooling
        # transpose to [batch,knum,dlen],then get max k in each kernal filter
        transpose_pos_gate_conv = tf.transpose(pos_gate_conv, perm=[0, 2, 1])
        transpose_pos_prf_1_gate_conv = tf.transpose(pos_prf_1_gate_conv, perm=[0, 2, 1])
        transpose_pos_prf_2_gate_conv = tf.transpose(pos_prf_2_gate_conv, perm=[0, 2, 1])
        transpose_pos_prf_3_gate_conv = tf.transpose(pos_prf_3_gate_conv, perm=[0, 2, 1])

        # [batch,k_num,maxpolling_num]
        pos_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_pos_gate_conv,
            k=self.maxpooling_num,
        )
        pos_prf_1_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_pos_prf_1_gate_conv,
            k=self.maxpooling_num,
        )
        pos_prf_2_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_pos_prf_2_gate_conv,
            k=self.maxpooling_num,
        )
        pos_prf_3_kmaxpooling, _ = tf.nn.top_k(
            input=transpose_pos_prf_3_gate_conv,
            k=self.maxpooling_num,
        )
        pos_encoder = tf.reshape(pos_kmaxpooling, shape=(self.batch_test_size, -1))
        pos_prf_1_encoder = tf.reshape(pos_prf_1_kmaxpooling, shape=(self.batch_test_size, -1))
        pos_prf_2_encoder = tf.reshape(pos_prf_2_kmaxpooling, shape=(self.batch_test_size, -1))
        pos_prf_3_encoder = tf.reshape(pos_prf_3_kmaxpooling, shape=(self.batch_test_size, -1))

        pos_all_encoder = tf.concat([pos_encoder, pos_prf_1_encoder, pos_prf_2_encoder, pos_prf_3_encoder], axis=1)


        pos_decoder_mlp1 = tf.layers.dense(
            inputs=pos_all_encoder,
            units=self.decoder_mlp1_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp1'
        )

        pos_decoder_mlp2 = tf.layers.dense(
            inputs=pos_decoder_mlp1,
            units=self.decoder_mlp2_num,
            activation=tf.nn.tanh,
            trainable=True,
            name='decoder_mlp2'
        )

        score_pos = pos_decoder_mlp2
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        save_var = [v for v in tf.trainable_variables()]
        # Create a local session to run the testing.
        for i in range(int(self.max_epochs / self.checkpoint_steps)):
            with tf.Session(config=config) as sess:
                test_point_stream = open(test_point_file_path)
                outfile = open(output_file_path + '-epoch' + str(self.checkpoint_steps * (i + 1)) + '.txt', 'w')
                saver = tf.train.Saver(var_list=save_var)

                if load_model:
                    p = checkpoint_dir + 'zsl' + str(self.checkpoint_steps * (i + 1)) + '.ckpt'
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    saver.restore(sess, p)
                    print("data loaded!")
                else:
                    init = tf.global_variables_initializer()
                    sess.run(init)

                # Loop through training steps.

                for b in range(int(np.ceil(float(test_size) / self.batch_test_size))):
                    X = next(self.test_data_generator.test_pairwise_reader(test_point_stream, self.batch_test_size))
                    if (X[u'q'].shape[0] != self.batch_test_size):
                        continue
                    query = X[u'q']
                    pos_doc = X[u'd']
                    prf_1 = X[u'prf_1']
                    prf_2 = X[u'prf_2']
                    prf_3 = X[u'prf_3']
                    test_q_lens = X[u'q_lens']
                    test_prf_1_lens = X[u'prf_1_lens']
                    test_prf_2_lens = X[u'prf_2_lens']
                    test_prf_3_lens = X[u'prf_3_lens']
                    M_query = self.gen_test_query_mask(query)
                    M_pos = self.gen_test_doc_mask(pos_doc)
                    M_prf_1 = self.gen_test_doc_mask(prf_1)
                    M_prf_2 = self.gen_test_doc_mask(prf_2)
                    M_prf_3 = self.gen_test_doc_mask(prf_3)
                    test_feed_dict = {input_q: query,
                                      input_pos_d: pos_doc,
                                      q_lens: test_q_lens,
                                      q_mask: M_query,
                                      pos_d_mask: M_pos,
                                      input_prf_1: prf_1,
                                      input_prf_2: prf_2,
                                      input_prf_3: prf_3,
                                      prf_1_lens: test_prf_1_lens,
                                      prf_2_lens: test_prf_2_lens,
                                      prf_3_lens: test_prf_3_lens,
                                      prf_1_mask: M_prf_1,
                                      prf_2_mask: M_prf_2,
                                      prf_3_mask: M_prf_3
                                      }

                    # Run the graph and fetch some of the nodes.
                    scores = sess.run(score_pos, feed_dict=test_feed_dict)

                    for score in scores:
                        outfile.write('{0}\n'.format(score[0]))

                outfile.close()
                test_point_stream.close()


if __name__ == '__main__':
    train_pair_file_path = './movie-review-5class/movie-review-0/movie-train.txt'
    val_pair_file_path = './movie-review-5class/movie-review-0/movie-val.txt'
    checkpoint_dir = './checkpoint/movie0/'

    test_file = './movie-review-5class/movie-review-0/movie-test.txt'
    test_size = 5675
    output_score_file = './output/movie0/movie'

    load_model = True

    Train = False

    if Train:
        nn = ZSL_TC()
        nn.train(train_pair_file_path=train_pair_file_path,
                 val_pair_file_path=val_pair_file_path,
                 checkpoint_dir=checkpoint_dir,
                 load_model=False)
    else:
        nn = ZSL_TC()
        nn.test(test_point_file_path=test_file,
                test_size=test_size,
                output_file_path=output_score_file,
                load_model=True,
                checkpoint_dir=checkpoint_dir)






