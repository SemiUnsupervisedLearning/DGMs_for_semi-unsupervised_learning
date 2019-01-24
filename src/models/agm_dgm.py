from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from models.model import model

import sys, os, pdb

import numpy as np
np.random.seed(1)
import utils.dgm as dgm 

import tensorflow as tf
tf.set_random_seed(1)
import keras

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, \
    Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Concatenate, \
    concatenate, Activation, RepeatVector, Dropout, Dot, dot, BatchNormalization
from keras.models import Sequential, Model
from keras.utils import np_utils, multi_gpu_model, print_summary
from keras import metrics, callbacks, initializers

from keras import backend as K

from tensorflow.contrib.tensorboard.plugins import projector


""" 
Implementation of semi-supervised DGMs from Kingma et al. (2014): p(z) * p(y) * p(x|y,z) 
Inference network: q(z,y|x) = q(y|x) * q(z|y,x) 
"""

class agm_dgm(model):
   
    def __init__(self, n_x, n_y, n_h, n_z=2,n_a=2, x_dist='Gaussian', mc_samples=1,alpha=0.1, l2_reg=0.3,learning_paradigm='supervised', name = 'model',ckpt=None, prior=None, output_dir=None, analytic_kl=False, loss_balance='average', model_name='agm_dgm'):
        self.reg_term = tf.placeholder(tf.float32, shape=[], name='reg_term')
        self.n_a = n_a        # auxiliary variable dimension
        if prior is None:
            self.prior = tf.constant(np.array([1.0/n_y]*n_y), dtype=tf.float32, shape=[1, n_y], name='prior_p_y')
        else:
            self.prior = tf.constant(prior, dtype=tf.float32, shape=[1, n_y], name='prior_p_y')

        super(agm_dgm, self).__init__(n_x, n_y, n_h, n_z, x_dist, mc_samples, l2_reg,
                                 alpha, ckpt, learning_paradigm, name,
                                 analytic_kl, output_dir, loss_balance, model_name)
        """ TODO: add any general terms we want to have here """


    def build_model(self):
        """ Define model components and variables """
        self.create_placeholders()
        #
        glorot_initializer = initializers.glorot_normal()
        normal_initializer = initializers.random_normal(stddev=0.001)
        self.q_y_ax_model=Sequential()
        self.q_y_ax_model.add(Dense(self.intermediate_dim,name='hidden_1',kernel_initializer=glorot_initializer, bias_initializer=normal_initializer,input_dim=self.n_x + self.n_a))
        self.q_y_ax_model.add(Activation('relu'))
        self.q_y_ax_model.add(Dense(self.intermediate_dim,name='hidden_2',kernel_initializer=glorot_initializer, bias_initializer=normal_initializer))
        self.q_y_ax_model.add(Activation('relu'))
        self.q_y_ax_model.add(Dense(self.n_y,name='q_y_xa_logit',kernel_initializer=initializers.random_normal(stddev=0.001), bias_initializer=normal_initializer))
        #
        self.q_z_axy = Sequential()
        self.q_z_axy.add(Dense(self.intermediate_dim,kernel_initializer=glorot_initializer, bias_initializer=normal_initializer, input_dim=self.n_a+self.n_x+self.n_y))
        self.q_z_axy.add(Activation('relu'))
        self.q_z_axy.add(Dense(self.intermediate_dim, kernel_initializer=glorot_initializer, bias_initializer=normal_initializer))
        self.q_z_axy.add(Activation('relu'))
        #
        self.q_z_axy_mean = Sequential()
        self.q_z_axy_mean.add(self.q_z_axy)
        self.q_z_axy_mean.add(Dense(self.n_z,kernel_initializer=normal_initializer, bias_initializer=normal_initializer))
        #
        self.q_z_axy_log_var = Sequential()
        self.q_z_axy_log_var.add(self.q_z_axy)
        self.q_z_axy_log_var.add(Dense(self.n_z,kernel_initializer=normal_initializer, bias_initializer=normal_initializer))
        #
        self.q_a_x_model=Sequential()
        self.q_a_x_model.add(Dense(self.intermediate_dim, kernel_initializer=glorot_initializer, bias_initializer=normal_initializer, input_dim=self.n_x))
        self.q_a_x_model.add(Activation('relu'))
        self.q_a_x_model.add(Dense(self.intermediate_dim,kernel_initializer=glorot_initializer, bias_initializer=normal_initializer))
        self.q_a_x_model.add(Activation('relu'))
        self.q_a_x_mean = Sequential()
        self.q_a_x_mean.add(self.q_a_x_model)
        self.q_a_x_mean.add(Dense(self.n_a,kernel_initializer=normal_initializer, bias_initializer=normal_initializer))
        self.q_a_x_log_var = Sequential()
        self.q_a_x_log_var.add(self.q_a_x_model)
        self.q_a_x_log_var.add(Dense(self.n_a,kernel_initializer=normal_initializer, bias_initializer=normal_initializer))
        #
        self.p_a_xyz_model=Sequential()
        self.p_a_xyz_model.add(Dense(self.intermediate_dim, kernel_initializer=glorot_initializer, bias_initializer=normal_initializer, input_dim=self.n_z+self.n_y))
        self.p_a_xyz_model.add(Activation('relu'))
        self.p_a_xyz_model.add(Dense(self.intermediate_dim,kernel_initializer=glorot_initializer, bias_initializer=normal_initializer))
        self.p_a_xyz_model.add(Activation('relu'))
        self.p_a_xyz_mean = Sequential()
        self.p_a_xyz_mean.add(self.p_a_xyz_model)
        self.p_a_xyz_mean.add(Dense(self.n_a,kernel_initializer=normal_initializer, bias_initializer=normal_initializer))
        self.p_a_xyz_log_var = Sequential()
        self.p_a_xyz_log_var.add(self.p_a_xyz_model)
        self.p_a_xyz_log_var.add(Dense(self.n_a,kernel_initializer=normal_initializer, bias_initializer=normal_initializer))

        self.p_z_y_mean=Sequential()
        self.p_z_y_mean.add(Dense(self.n_z,input_dim=self.n_y,kernel_initializer=normal_initializer, bias_initializer='zeros'))
        self.p_z_y_log_var=Sequential()
        self.p_z_y_log_var.add(Dense(self.n_z,input_dim=self.n_y,kernel_initializer=glorot_initializer, bias_initializer='zeros'))
        #
        if self.x_dist == 'Gaussian':
            self.p_x_yz_model=Sequential()
            self.p_x_yz_model.add(Dense(self.intermediate_dim, kernel_initializer=glorot_initializer, bias_initializer=normal_initializer, input_dim=self.n_z))
            self.p_x_yz_model.add(Activation('relu'))
            self.p_x_yz_model.add(Dense(self.intermediate_dim,kernel_initializer=glorot_initializer, bias_initializer=normal_initializer))
            self.p_x_yz_model.add(Activation('relu'))
            self.p_x_yz_mean = Sequential()
            self.p_x_yz_mean.add(self.p_x_yz_model)
            self.p_x_yz_mean.add(Dense(self.n_x,kernel_initializer=normal_initializer, bias_initializer=normal_initializer))
            self.p_x_yz_log_var = Sequential()
            self.p_x_yz_log_var.add(self.p_x_yz_model)
            self.p_x_yz_log_var.add(Dense(self.n_x,kernel_initializer=normal_initializer, bias_initializer=normal_initializer))
            #
        elif self.x_dist == 'Bernoulli':
            self.p_x_yz_mean_model=Sequential()
            self.p_x_yz_mean_model.add(Dense(self.intermediate_dim, kernel_initializer=glorot_initializer, bias_initializer=normal_initializer, input_dim=self.n_z))
            self.p_x_yz_mean_model.add(Activation('relu'))
            self.p_x_yz_mean_model.add(Dense(self.intermediate_dim,kernel_initializer=glorot_initializer, bias_initializer=normal_initializer))
            self.p_x_yz_mean_model.add(Activation('relu'))
            self.p_x_yz_mean_model.add(Dense(self.n_x, kernel_initializer=initializers.random_normal(stddev=0.001), bias_initializer=initializers.random_normal(stddev=0.001)))

    def compute_loss(self):
        """ manipulate computed components and compute loss """
        #note: ELBO_l now includes qy_ll loss
        elbo_l, qy_l = self.labelled_loss(self.x_l, self.y_l)
        self.elbo_l = tf.reduce_mean(elbo_l)
        self.qy_ll = tf.reduce_mean(qy_l)
        self.elbo_u = tf.reduce_mean(self.unlabelled_loss(self.x_u))
        weight_priors = self.l2_reg*self.weight_prior()/self.reg_term
        if self.loss_balance == 'average':
            return -(self.elbo_l + self.elbo_u + self.qy_ll + weight_priors)
        elif self.loss_balance == 'weighted':
            return -((float(self.n_l)/float(self.n_train)) * self.elbo_l + (float(self.n_u)/float(self.n_train)) * self.elbo_u + self.qy_ll + weight_priors)

    def compute_unsupervised_loss(self):
        """ manipulate computed components and compute unsup loss """
        self.elbo_u = tf.reduce_mean(self.unlabelled_loss(self.x_u))
        weight_priors = self.l2_reg*self.weight_prior()/self.reg_term   
        return -(self.elbo_u + weight_priors)

    def compute_supervised_loss(self):
        """ manipulate computed components and compute loss """
        #note: ELBO_l now includes qy_ll loss
        elbo_l, qy_l = self.labelled_loss(self.x_l, self.y_l)
        self.elbo_l = tf.reduce_mean(elbo_l)
        self.qy_ll = tf.reduce_mean(qy_l)
        weight_priors = self.l2_reg*self.weight_prior()/self.reg_term   
        return -(self.elbo_l + self.qy_ll + weight_priors)

    def labelled_loss(self, x, y):
        a_m, a_lv, a = self.sample_a(x)
        a_m, a_lv = tf.tile(tf.expand_dims(a_m,0), [self.mc_samples,1,1]), tf.tile(tf.expand_dims(a_lv,0),[self.mc_samples,1,1])
        x_ = tf.tile(tf.expand_dims(x, 0), [self.mc_samples, 1,1])
        y_ = tf.tile(tf.expand_dims(y,0),[self.mc_samples,1,1])
        z_m, z_lv, z = self.sample_z(x_, y_, a)
        z_m_p, z_lv_p = self.calc_z_prior(y_)
        l_loss = self.lowerBound(x_, y_, z, z_m, z_lv, a, a_m, a_lv, z_m_p, z_lv_p)
        qy_loss = tf.reduce_mean(self.qy_loss(x_,y_,a, expand_y=False) * self.alpha, axis=0)
        return l_loss, qy_loss

    def unlabelled_loss(self, x):
        a_m, a_lv, a = self.sample_a(x)
        qy_l = self.predict(x,a)
        x_r, a_r = tf.tile(x, [self.n_y,1]), tf.tile(a, [1,self.n_y,1])
        a_mr, a_lvr = tf.tile(tf.expand_dims(a_m,0), [1,self.n_y,1]), tf.tile(tf.expand_dims(a_lv,0), [1,self.n_y,1])
        y_u = tf.reshape(tf.tile(tf.eye(self.n_y), [1, tf.shape(x)[0]]), [-1, self.n_y])
        x_ = tf.tile(tf.expand_dims(x_r,0), [self.mc_samples, 1,1])
        y_ = tf.tile(tf.expand_dims(y_u,0), [self.mc_samples, 1,1])
        z_m, z_lv, z = self.sample_z(x_, y_, a_r)
        z_m_p, z_lv_p = self.calc_z_prior(y_)
        n_u = tf.shape(x)[0] 
        lb_u = tf.transpose(tf.reshape(self.lowerBound(x_, y_, z, z_m, z_lv, a_r, a_mr, a_lvr, z_m_p, z_lv_p), [self.n_y, n_u]))
        lb_u = tf.reduce_sum(qy_l * lb_u, axis=-1)
        qy_entropy = -tf.reduce_sum(qy_l * tf.log(qy_l + 1e-10), axis=-1)
        return lb_u + qy_entropy

    def lowerBound(self, x, y, z, z_m, z_lv, a, a_m, a_lv, z_m_p, z_lv_p):
        """ Compute densities and lower bound given all inputs (mc_samps X n_obs X n_dim) """
        pa_in = tf.reshape(tf.concat([y, z], axis=-1), [-1, self.n_y + self.n_z])
        a_m_p, a_lv_p = self.p_a_xyz_mean(pa_in), self.p_a_xyz_log_var(pa_in)
        a_m_p, a_lv_p = tf.reshape(a_m_p, [self.mc_samples,-1,self.n_a]), tf.reshape(a_lv_p, [self.mc_samples,-1,self.n_a])
        l_px = self.compute_logpx(x,z)
        l_py = dgm.multinoulliLogDensity(y,self.prior,on_priors=True)
        l_pz = dgm.gaussianLogDensity(z, z_m_p, z_lv_p)
        l_pa = dgm.gaussianLogDensity(a, a_m_p, a_lv_p)
        l_qz = dgm.gaussianLogDensity(z, z_m, z_lv)
        l_qa = dgm.gaussianLogDensity(a, a_m, a_lv)
        return tf.reduce_mean(l_px + l_py + l_pz + l_pa - l_qz - l_qa, axis=0)  
    
    def qy_loss(self, x, y=None, a=None, expand_y = True):
        if a is None:
            _, _, a = self.sample_a(x)
            qy_in = tf.reshape(tf.concat([tf.tile(tf.expand_dims(x,0), [self.mc_samples,1,1]), a], axis=-1), [-1, self.n_x+self.n_a])
        else:
            qy_in = tf.reshape(tf.concat([x, a], axis=-1), [-1, self.n_x+self.n_a])
        y_ = tf.reshape(self.q_y_ax_model(qy_in),  [self.mc_samples,-1,self.n_y])
        if y is not None and expand_y == True:
            y = tf.tile(tf.expand_dims(y,0), [self.mc_samples, 1,1])
        if y is None:
            return dgm.multinoulliUniformLogDensity(y_)
        else:
            return dgm.multinoulliLogDensity(y, y_)


    def sample_z(self, x, y, a, n_samples=None):
        if n_samples == None:
            n_samples = 1
        l_qz_in = tf.reshape(tf.concat([x, y, a], axis=-1), [-1, self.n_x + self.n_y + self.n_a])
        z_mean = dgm.forwardPass(self.q_z_axy_mean,l_qz_in)
        z_log_var = dgm.forwardPass(self.q_z_axy_log_var,l_qz_in)
        z = dgm.sampleNormal(z_mean, z_log_var, mc_samps=n_samples)
        z_mean, z_log_var = tf.reshape(z_mean, [self.mc_samples,-1,self.n_z]), tf.reshape(z_log_var, [self.mc_samples,-1,self.n_z])
        z = tf.reshape(z, [self.mc_samples,-1,self.n_z])
        return z_mean, z_log_var, z

    def sample_a(self, x, n_samples=None):
        if n_samples == None:
            n_samples = self.mc_samples
        l_qa_in = x
        a_mean = dgm.forwardPass(self.q_a_x_mean,l_qa_in)
        a_log_var = dgm.forwardPass(self.q_a_x_log_var,l_qa_in)
        return a_mean, a_log_var, dgm.sampleNormal(a_mean, a_log_var,n_samples)

    def calc_z_prior(self, y):
        z_mean_prior = dgm.forwardPass(self.p_z_y_mean,y)
        z_log_var_prior = dgm.forwardPass(self.p_z_y_log_var,y)
        z_mean_prior, z_log_var_prior = tf.reshape(z_mean_prior, [self.mc_samples,-1,self.n_z]), tf.reshape(z_log_var_prior, [self.mc_samples,-1,self.n_z])
        return z_mean_prior, z_log_var_prior

    def compute_logpx(self, x, z):
        px_in = tf.reshape(z, [-1, self.n_z])
        if self.x_dist == 'Gaussian':
            mean, log_var = self.p_x_yz_mean(px_in), self.p_x_yz_log_var(px_in)
            mean, log_var = tf.reshape(mean, [self.mc_samples, -1, self.n_x]),  tf.reshape(log_var, [self.mc_samples, -1, self.n_x])
            return dgm.gaussianLogDensity(x, mean, log_var)
        elif self.x_dist == 'Bernoulli':
            logits = self.p_x_yz_mean_model(px_in)
            logits = tf.reshape(logits, [self.mc_samples, -1, self.n_x])
            return dgm.bernoulliLogDensity(x, logits) 

    def predict(self, x, a=None, n_iters=None):
        if n_iters == None:
            n_iters=self.mc_samples
        if a is None:
            _, _, a = self.sample_a(x, n_iters)
        qy_in = tf.reshape(tf.concat([tf.tile(tf.expand_dims(x,0), [n_iters,1,1]), a], axis=-1), [-1, self.n_x+self.n_a])
        """ predict y for given x, a with q(y|x, a) """
        y_ = tf.reshape(self.q_y_ax_model(qy_in),  [n_iters,-1,self.n_y])
        return tf.reduce_mean(tf.nn.softmax(y_), axis=0)


    def encode(self, x, y=None, n_iters=1):
        """ encode a new example into z-space (labelled or unlabelled) """
        if y is None:
            y = tf.one_hot(tf.argmax(self.predict(x, n_iters)))
        _, _, a = self.sample_a(x, n_iters)
        z_mean, z_log_var, z = self.sample_z(x, y, a, n_iters)
        return z_mean, z_log_var, z

    #def training_fd(self, x_l, y_l, x_u):
    #    return {self.x_l: x_l, self.y_l: y_l, self.x_u: x_u, self.x: x_l, self.y: y_l, self.reg_term:self.n_train}

    def _printing_feed_dict(self, Data, x_l, x_u, y, eval_samps, binarize):
        fd = super(agm_dgm,self)._printing_feed_dict(Data, x_l, x_u, y, eval_samps, binarize)
        fd[self.reg_term] = self.n_train
        return fd

    def print_verbose1(self, epoch, fd, sess):
        total, elbo_l, elbo_u, qy_ll, weight_priors = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u, self.qy_ll, weight_priors] ,fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)     
        print("Epoch: {}: Total: {:5.3f}, labelled: {:5.3f}, Unlabelled: {:5.3f}, q_y_ll: {:5.3f}, weight_priors: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, qy_ll, weight_priors, train_acc, test_acc))  

    def print_verbose2(self, epoch, fd, sess):
        total, elbo_l, elbo_u = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u] ,fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)     
        print("Epoch: {}: Total: {:5.3f}, labelled: {:5.3f}, Unlabelled: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, train_acc, test_acc))

    def print_verbose3(self, epoch):
        print("Epoch: {}: Total: {:5.3f}, Unlabelled: {:5.3f}, KL_y: {:5.3f}, TrainingAc: {:5.3f}, TestingAc: {:5.3f}, TrainingK: {:5.3f}, TestingK: {:5.3f}".format(epoch, sum(self.curve_array[epoch][1:3]), self.curve_array[epoch][2], self.curve_array[epoch][3], self.curve_array[epoch][0], self.curve_array[epoch][6], self.curve_array[epoch][12], self.curve_array[epoch][13])) 
