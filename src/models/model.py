from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import os
import numpy as np
np.random.seed(1)
import utils.dgm as dgm
from scipy.stats import mode
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import tensorflow as tf
tf.set_random_seed(1)
from tensorflow.contrib.tensorboard.plugins import projector
from utils.checkmate import BestCheckpointSaver, get_best_checkpoint
from keras import backend as K
import re
""" Super class for deep generative models """

class model(object):

    def __init__(self, n_x, n_y, n_h, n_z=2, x_dist='Gaussian', mc_samples=1,
                 alpha=0.1, l2_reg=0.3, ckpt=None, learning_paradigm='supervised',
                 name='model', analytic_kl=False, output_dir=None,
                 loss_balance='average', model_name='dgm', decay_period=200,
                 decay_ratio=0.75):

        self.n_x, self.n_y = n_x, n_y    # data characteristics
        self.n_y_tf = tf.constant(float(self.n_y))
        self.n_z = n_z                   # number of labelled dimensions
        self.x_dist = x_dist             # likelihood for inputs
        self.mc_samples = mc_samples     # MC samples for estimation
        self.mc_samples_tf = tf.constant(float(self.mc_samples))
        self.alpha = alpha               # additional penalty weight term
        self.l2_reg = l2_reg             # weight regularization scaling const.
        self.name = name              # model name
        self.ckpt = ckpt                 # preallocated checkpoint dir
        self.intermediate_dim = n_h
        self.n, self.n_train, self.n_u, self.n_l = 1, 1, 1, 1       # initialize data size
        self.output_dir = output_dir

        self.build_model()
        self.learning_paradigm = learning_paradigm
        # if self.learning_paradigm == 'semisupervised' or 'un-semisupervised':
        #     self.loss = self.compute_loss()
        # if self.learning_paradigm == 'unsupervised':
        #     self.loss = self.compute_unsupervised_loss()
        # if self.learning_paradigm == 'supervised':
        #     self.loss = self.compute_supervised_loss()
        self.session = tf.Session()
        self.predictions = self.predict(self.x)
        self.loss_balance = loss_balance
        #^ How to sum unlabelled and labelled loss terms.
        # average: add both as averages over each of their minibatches
        # weighted: weight in proportion to the sizes of the lab and unlab datasets.
        self.model_name = model_name
        self.decay_period = decay_period
        self.decay_ratio = decay_ratio

    def train(self, Data, n_epochs, l_bs, u_bs, lr, eval_samps=None,
              binarize=False, verbose=1, decay_ratio=0.75, decay_period=200,
              h_opt=False, keep_ckpt=True, restore=False):
        """ Method for training the models """
        self.data_init(Data, eval_samps, l_bs, u_bs)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
        self.epoch = 0
        #self.lr = self.set_learning_rate([lr[0], 1600, lr[0] / 10.0])
        self.lr = self.set_learning_rate([lr[0], lr[0] / 10.0, decay_period, decay_ratio], 'exp')
        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gvs = optimizer.compute_gradients(self.loss)
        # clip gradients
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                      for grad, var in gvs]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = optimizer.apply_gradients(
                capped_gvs, global_step=self.global_step)

        self.y_pred = self.predict(self.x)
        self.curve_array = np.zeros((n_epochs + 1, 14))
        if self.learning_paradigm == 'unsupervised':
            self.elbo_l_curve = tf.reduce_mean(
                self.unlabelled_loss(self.x))
            self.qy_ll_curve = tf.reduce_mean(
                self.qy_loss(self.x))
            self.elbo_u_curve = tf.reduce_mean(
                self.unlabelled_loss(self.x))
        else:
            if self.model_name == 'adgm' or self.model_name == 'adg_dgm':
                self.elbo_l_curve = tf.reduce_mean(
                    self.labelled_loss(self.x, self.y)[0])
                self.qy_ll_curve = tf.reduce_mean(
                    self.labelled_loss(self.x, self.y)[1])
            else:
                self.elbo_l_curve = tf.reduce_mean(
                    self.labelled_loss(self.x, self.y))
                self.qy_ll_curve = tf.reduce_mean(
                    self.qy_loss(self.x, self.y))

            self.elbo_u_curve = tf.reduce_mean(
                self.unlabelled_loss(self.x))

        self.compute_accuracies()


        # initialize session and train
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            if restore == True:
                saver_for_restore = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
                best_ckpt = get_best_checkpoint(self.ckpt_dir)
                best_epoch = int(re.match('.*?([0-9]+)$', best_ckpt).group(1))
                best_ckpt_usable = re.sub('-([0-9]+)$', "", best_ckpt)
                saver_for_restore.restore(sess, best_ckpt_usable)
                self.epoch = best_epoch
            self.curve_array[self.epoch] = self.calc_curve_vals(sess, Data)
            if verbose == 3:
                self.print_verbose3(self.epoch)
            if keep_ckpt == True:
                saver = BestCheckpointSaver(save_dir=self.ckpt_dir,
                                        num_to_keep=2,
                                        maximize=True)
            while self.epoch < n_epochs:

                x_labelled, labels, x_unlabelled, _ = \
                    Data.next_batch(l_bs, u_bs)

                if binarize is True:
                    x_labelled = self.binarize(x_labelled)
                    x_unlabelled = self.binarize(x_unlabelled)

                fd = self.training_fd(x_labelled, labels, x_unlabelled)
                _, loss_batch = sess.run([self.optimizer, self.loss], fd)

                if Data._epochs_unlabelled > self.epoch:
                    self.epoch += 1
                    # sess.run(self.global_epoch.assign(self.epoch)
                    self.curve_array[self.epoch] = \
                        self.calc_curve_vals(sess, Data)
                    if h_opt == True and self.epoch > 20:
                        if self.curve_array[self.epoch, 12] < 0.07:
                            raise Exception('results too bad')
                    if h_opt == True and self.epoch > 40:
                        if self.curve_array[self.epoch, 12] < 0.1:
                            raise Exception('results too bad')
                    if keep_ckpt == True:
                        saver.handle(self.curve_array[self.epoch, 6],
                                    sess, self.global_step, self.epoch)
                    if verbose == 1:
                        fd = self._printing_feed_dict(Data, x_labelled,
                                                      x_unlabelled, labels,
                                                      eval_samps, binarize)
                        self.print_verbose1(self.epoch, fd, sess)
                    elif verbose == 2:
                        fd = self._printing_feed_dict(Data, x_labelled,
                                                      x_unlabelled, labels,
                                                      eval_samps, binarize)
                        self.print_verbose2(self.epoch, fd, sess)
                    elif verbose == 3:
                        self.print_verbose3(self.epoch)
                        if self.epoch % 10 == 0:
                            y_pred_test = sess.run([self.y_pred],
                                                   {self.x: Data.data['x_test'],
                                                    K.learning_phase(): 0})[0]
                            conf_mat = confusion_matrix(
                                Data.data['y_test'].argmax(1),
                                y_pred_test.argmax(1))
                            np.save(os.path.join(
                                    self.output_dir,
                                    'conf_mat_' + self.name + '_' + str(self.epoch) + '.npy'),
                                    conf_mat)
                            np.save(os.path.join(
                                    self.output_dir,
                                    'y_pred_' + self.name + '_' + str(self.epoch) + '.npy'),
                                     y_pred_test)
                            np.save(os.path.join(
                                    self.output_dir,
                                    'y_true_' + self.name + '_' + str(self.epoch) + '.npy'),
                                    Data.data['y_test'])
                    if np.sum(np.isnan(self.curve_array)) > 0:
                        print('loss is nan, going back to previous best checkpoint')
                        best_ckpt = get_best_checkpoint(self.ckpt_dir)
                        best_epoch = int(re.match('.*?([0-9]+)$', best_ckpt).group(1))
                        best_ckpt_usable = re.sub('-([0-9]+)$', "", best_ckpt)
                        self.epoch = best_epoch
                        saver._saver.restore(sess, best_ckpt_usable)
        return self.curve_array

    def calc_curve_vals(self, sess, Data, train_batch_size=1000):
        """ Function to calculate output arary row during training

        return: an ndarray row with entries:
        [0] - training accuracy
        [1] - ELBO on labelled training data
        [2] - ELBO on unlabelled training data
        [3] - Cross entropy loss on labelled training data
        [4] - scaled weight prior value on training data
        [5] - Total train data loss - ELBO u + ELBO l + alpha * X ent loss

        [6:11] - the same as [0:5] for but for test data
        [12] - Cohen's Kappa for training data
        [13] - Cohen's Kappa for test data

        Parameters
        ----------
        sess tf.session: current training session
        Data custom data class: Data being used for training and testing
        train_batch_size int: number of data points to use to evaluate [0:5]
        """

        output_array = np.zeros((14))
        x_train_sample, y_train_sample = Data.sample_train(train_batch_size)
        sess_run_outputs = sess.run([self.y_pred, self.elbo_l_curve,
                                     self.elbo_u_curve, self.qy_ll_curve,
                                     self.l2_reg * self.weight_prior() /
                                     self.reg_term],
                                    {self.x: x_train_sample,
                                     self.y: y_train_sample,
                                     self.reg_term: self.n_train,
                                     K.learning_phase(): 0})

        output_array[0] = self.compute_acc(sess_run_outputs[0],
                                           y_train_sample,
                                           self.learning_paradigm)

        output_array[1:5] = np.array(sess_run_outputs[1:])

        output_array[5] = output_array[1] + output_array[2] + \
            self.alpha_np * output_array[3] + output_array[4]

        output_array[12] = cohen_kappa_score(self.compute_pred(
                                             sess_run_outputs[0],
                                             y_train_sample),
                                             np.argmax(y_train_sample, axis=-1))

        sess_run_outputs = sess.run([self.y_pred, self.elbo_l_curve,
                                     self.elbo_u_curve, self.qy_ll_curve,
                                     self.l2_reg * self.weight_prior() /
                                     self.reg_term],
                                    {self.x: Data.data['x_test'][0:1000],
                                     self.y: Data.data['y_test'][0:1000],
                                     self.reg_term: self.n_train,
                                     K.learning_phase(): 0})

        output_array[6] = self.compute_acc(sess_run_outputs[0],
                                           Data.data['y_test'][0:1000],
                                           self.learning_paradigm)

        output_array[7:11] = np.array(sess_run_outputs[1:])

        output_array[11] = output_array[7] + output_array[8] + \
            self.alpha_np * output_array[8] + output_array[10]

        output_array[13] = cohen_kappa_score(self.compute_pred(
                                             sess_run_outputs[0],
                                             Data.data['y_test'][0:1000]),
                                             np.argmax(Data.data['y_test'][0:1000],
                                                       axis=-1))
        return output_array

    def encode_new(self, x, y=None):
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            encoded = session.run([self.encode(self.x, self.y)], {self.x: x, self.y: y})
        return encoded

    def predict_new(self, x):
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            preds = session.run(self.predictions, {self.x: x})
        return preds

    def generate_new(self, n_samps, y=None, z=None):
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            data = session.run(self.generateX(n_samps, y, z))
        return data

    def return_z_prior(self, y = None):
        if y is None:
            y = np.eye(self.n_y)
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            data = session.run(self.calc_z_prior(self.y), {self.y:y})
        return data



### Every instance of model must implement these two methods ###

    def predict(self, x):
        pass

    def encode(self, x):
        pass

    def build_model(self):
        pass

    def compute_loss(self):
        pass

    def compute_unsupervised_loss(self):
        pass

    def compute_supervised_loss(self):
        pass

################################################################

    def weight_prior(self):
        weights = [V for V in tf.trainable_variables()
                   if 'W' in V.name or 'kernel' in V.name]
        return np.sum([tf.reduce_sum(dgm.standardNormalLogDensity(w))
                       for w in weights])

    def weight_regularization(self):
        weights = [V for V in tf.trainable_variables() if 'W' in V.name]
        return np.sum([tf.nn.l2_loss(w) for w in weights])

    def data_init(self, Data, eval_samps, l_bs, u_bs):
        self._process_data(Data, eval_samps, l_bs, u_bs)

    def binarize(self, x):
        return np.random.binomial(1, x)

    def set_schedule(self, temp_epochs, start_temp, n_epochs):
        if not temp_epochs:
            return np.ones((n_epochs, )).astype('float32')
        else:
            warmup = np.expand_dims(np.linspace(start_temp,
                                                1.0, temp_epochs), 1)
            plateau = np.ones((n_epochs - temp_epochs, 1))
            return np.ravel(np.vstack((warmup, plateau))).astype('float32')

    def set_learning_rate(self, lr, mode=None):
        """ Set learning rate """
        if len(lr) == 1:
            return lr[0]
        elif mode=='poly':
            start_lr, rate, final_lr = lr
            return tf.train.polynomial_decay(start_lr,
                                             self.global_step,
                                             rate, end_learning_rate=final_lr)
        elif mode=='exp':
            start_lr, final_lr, num_epoch, ratio = lr
            lr_decay = tf.train.exponential_decay(start_lr, self.epoch, num_epoch,
                                                  ratio, staircase=True)
            return tf.maximum(lr_decay, tf.constant(final_lr))

    def _process_data(self, data, eval_samps, l_bs, u_bs):
        """ Extract relevant information from data_gen """
        self.n = data.N
        self.n_train = data.TRAIN_SIZE          # training set size
        self.n_test = data.TEST_SIZE            # test set size
        if eval_samps is None:
            self.eval_samps = self.n_train      # evaluation training set size
            self.eval_samps_test = self.n_test  # evaluation test set size
        else:
            self.eval_samps_train = eval_samps
            self.eval_samps_test = eval_samps
        self.n_l = data.NUM_LABELLED             # no. of labelled instances
        self.n_u = data.NUM_UNLABELLED           # no. of unlabelled instances
        self.data_name = data.NAME              # dataset being used
        self._allocate_directory()              # logging directory

        # alpha weighting for additional supervised loss term
        self.alpha_np = self.alpha * self.n_train / self.n_l
        self.alpha *= tf.constant(self.n_train / self.n_l)

    def create_placeholders(self):
        """ Create input/output placeholders """
        self.x_l = tf.placeholder(tf.float32, shape=[None, self.n_x],
                                  name='x_labelled')

        self.x_u = tf.placeholder(tf.float32, shape=[None, self.n_x],
                                  name='x_unlabelled')

        self.y_l = tf.placeholder(tf.float32, shape=[None, self.n_y],
                                  name='y_labelled')

        self.x_train = tf.placeholder(tf.float32, shape=[None, self.n_x],
                                      name='x_train')

        self.x_test = tf.placeholder(tf.float32, shape=[None, self.n_x],
                                     name='x_test')

        self.x = tf.placeholder(tf.float32, shape=[None, self.n_x],
                                name='x')

        self.y_train = tf.placeholder(tf.float32, shape=[None, self.n_y],
                                      name='y_train')

        self.y_test = tf.placeholder(tf.float32, shape=[None, self.n_y],
                                     name='y_test')

        self.y = tf.placeholder(tf.float32, shape=[None, self.n_y],
                                name='y')


    def compute_accuracies(self):
        self.train_acc = self.compute_acc_tf(self.x_train, self.y_train)
        self.test_acc = self.compute_acc_tf(self.x_test, self.y_test)

    def compute_acc_tf(self, x, y):
        y_ = self.predict(x)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, axis=1),
                                              tf.argmax(y, axis=1)),
                                     tf.float32))
        return acc

    def compute_acc(self, y, y_true, learning_paradigm):
        if learning_paradigm == 'unsupervised' \
           or learning_paradigm == 'semi-unsupervised':
            logits = y
            cat_pred = logits.argmax(1)
            real_pred = np.zeros_like(cat_pred)
            for cat in xrange(logits.shape[1]):
                idx = cat_pred == cat
                lab = y_true.argmax(1)[idx]
                if len(lab) == 0:
                    continue
                real_pred[cat_pred == cat] = mode(lab).mode[0]
            return np.mean(real_pred == y_true.argmax(1))
        elif learning_paradigm == 'supervised' or learning_paradigm == 'semisupervised':
            return np.mean(y.argmax(1) == y_true.argmax(1))

    def training_fd(self, x_l, y_l, x_u):
        if self.learning_paradigm == 'semisupervised' or self.learning_paradigm == 'semi-unsupervised':
            return {self.x_l: x_l, self.y_l: y_l, self.x_u: x_u, self.x: x_l, self.y: y_l, self.reg_term:self.n_train, K.learning_phase(): 1}
        if self.learning_paradigm == 'supervised':
            return {self.x_l: x_l, self.y_l: y_l, self.x: x_l, self.y: y_l, self.reg_term:self.n_train, K.learning_phase(): 1}
        if self.learning_paradigm == 'unsupervised':
            return {self.x_u: x_u, self.x: x_l, self.y: y_l, self.reg_term:self.n_train, K.learning_phase(): 1}

    def _printing_feed_dict(self, Data, x_l, x_u, y, eval_samps, binarize):
        x_train, y_train = Data.sample_train(eval_samps)
        x_test, y_test = Data.sample_test(eval_samps)
        return {self.x_train: x_train, self.y_train: y_train,
                self.x_test: x_test, self.y_test: y_test,
                self.x_l: x_l, self.y_l: y, self.x_u: x_u,
                K.learning_phase(): 0}

    def _allocate_directory(self):
        if self.ckpt is None:
            self.LOGDIR = os.path.join(self.output_dir, 'graphs', self.name)
            self.ckpt_dir = os.path.join(self.output_dir, 'ckpt', self.name)
        else:
            self.LOGDIR = os.path.join(self.output_dir, 'graphs', self.ckpt)
            self.ckpt_dir = os.path.join(self.output_dir, 'ckpt', self.ckpt)
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        if not os.path.isdir(self.LOGDIR):
            os.makedirs(self.LOGDIR)

    def compute_pred(self, y, y_true):
        logits = y
        cat_pred = logits.argmax(1)
        real_pred = np.zeros_like(cat_pred)
        for cat in xrange(logits.shape[1]):
            idx = cat_pred == cat
            lab = y_true.argmax(1)[idx]
            if len(lab) == 0:
                continue
            real_pred[cat_pred == cat] = mode(lab).mode[0]
        return real_pred