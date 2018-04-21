#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: kalman_norm.py
# Author: Guangrun Wang, Jiefeng Peng

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

from tensorpack.utils import logger
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.collection import backup_collection, restore_collection
from tensorpack.models.common import layer_register, VariableHolder

__all__ = ['KalmanNorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4

def get_bn_variables(n_out, use_scale, use_bias, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer())
    else:
        beta = tf.zeros([n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([n_out], name='gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(1.0), trainable=False)
    return beta, gamma, moving_mean, moving_var


def update_bn_ema(xn, batch_mean, batch_var,
                  moving_mean, moving_var, decay, internal_update):
    # TODO is there a way to use zero_debias in multi-GPU?
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')

    if internal_update:
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')
    else:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        return xn


def reshape_for_bn(param, ndims, chan, data_format):
    if ndims == 2:
        shape = [1, chan]
    else:
        shape = [1, 1, 1, chan] if data_format == 'NHWC' else [1, chan, 1, 1]
    return tf.reshape(param, shape)

def kalman_filter(x, pre_mean, pre_var, beta, gamma, shape, epsilon, split_num=1, p_rate = 0.0):
    inputs = tf.concat(tf.split(x, split_num, 0), -1) # N/split_num x H x W x C*split_num
    axis = [0, 1, 2]
    mean, variance = tf.nn.moments(inputs, axis) # C*split_num
    beta_, gamma_ = None, None
    beta_ = tf.reshape([beta]*split_num, [-1])
    gamma_ = tf.reshape([gamma]*split_num, [-1])

    flag = tf.reduce_mean(pre_var / variance)
    keep_fn = lambda: (mean, variance)
    def pre_updates():
        mean_ = p_rate * pre_mean + (1 - p_rate) * mean
        variance_ = p_rate * pre_var + (1 - p_rate) * variance  + p_rate * (1 - p_rate) * tf.square(mean - pre_mean)
        # mean_ = 0.1 * pre_mean + 0.9 * mean
        # variance_ = 0.1 * pre_var + 0.9 * variance  + 0.09 * tf.square(mean - pre_mean)
        return mean_, variance_
    mean, variance = tf.cond(flag < 100, pre_updates, keep_fn)
    # mean = 0.9 * pre_mean + 0.1 * mean
    # variance = 0.9 * pre_var + 0.1 * variance  + 0.09 * tf.square(mean - pre_mean)
    outputs = tf.nn.batch_normalization(inputs, mean, variance, beta_, gamma_, epsilon)
    outputs = tf.concat(tf.split(outputs, split_num, 3), 0)
    return outputs, mean, variance


@layer_register()
def KalmanNorm(x, pre_mean, pre_var, use_local_stat=None, decay=0.9, epsilon=1e-5,
              use_scale=True, use_bias=True,
              gamma_init=tf.constant_initializer(1.0), data_format='NHWC',
              internal_update=False, split_num = 1, p_rate = 0.0):
    """
    """    
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    if ndims == 2:
        data_format = 'NHWC'
    if data_format == 'NCHW':
        n_out = shape[1]
    else:
        n_out = shape[-1]  # channel
    assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
    beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_scale, use_bias, gamma_init)

    ctx = get_current_tower_context()
    if use_local_stat is None:
        use_local_stat = ctx.is_training
    use_local_stat = bool(use_local_stat)
    
    #batch_mean = None
    #batch_var = None
    if use_local_stat:
        if ndims == 2:
            x = tf.reshape(x, [-1, 1, 1, n_out])    # fused_bn only takes 4D input
            # fused_bn has error using NCHW? (see #190)

        if pre_mean is None and pre_var is None:
            inputs = tf.concat(tf.split(x, split_num, 0), -1) # N/S_n x H x W x C*S_n
            beta_, gamma_ = None, None
            beta_ = tf.reshape([beta]*split_num, [-1])
            gamma_ = tf.reshape([gamma]*split_num, [-1])
            xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
                inputs, gamma_, beta_, epsilon=epsilon,
                is_training=True, data_format=data_format)
            xn = tf.concat(tf.split(xn, split_num, 3), 0)
        else:
            xn, batch_mean, batch_var = kalman_filter(x, pre_mean, pre_var, beta, gamma, shape, epsilon, split_num, p_rate)
        #print(batch_var)
        #print(batch_mean)

        if ndims == 2:
            xn = tf.squeeze(xn, [1, 2])
    else:
        if ctx.is_training:
            assert get_tf_version_number() < 1.4, \
                "Fine tuning a BatchNorm model with fixed statistics is only " \
                "supported after https://github.com/tensorflow/tensorflow/pull/12580 "
            if ctx.is_main_training_tower:  # only warn in first tower
                logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
            # Using moving_mean/moving_variance in training, which means we
            # loaded a pre-trained BN and only fine-tuning the affine part.
            xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
                x, gamma, beta,
                mean=moving_mean, variance=moving_var, epsilon=epsilon,
                data_format=data_format, is_training=False)
        else:
            #if pre_mean is None and pre_var is None:
            #    inputs = tf.concat(tf.split(x, split_num, 0), -1) # N/S_n x H x W x C*S_n
            #    beta_, gamma_ = None, None
            #    beta_ = tf.reshape([beta]*split_num, [-1])
            #    gamma_ = tf.reshape([gamma]*split_num, [-1])
            #    xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
            #        inputs, gamma_, beta_, epsilon=epsilon,
            #        is_training=True, data_format=data_format)
            #    xn = tf.concat(tf.split(xn, split_num, 3), 0)
            #else:
            #    xn, batch_mean, batch_var = kalman_filter(x, pre_mean, pre_var, beta, gamma, shape, epsilon, split_num)

            # non-fused op is faster for inference  # TODO test if this is still true
            if ndims == 4 and data_format == 'NCHW':
                [g, b, mm, mv] = [reshape_for_bn(_, ndims, n_out, data_format)
                                  for _ in [gamma, beta, moving_mean, moving_var]]
                xn = tf.nn.batch_normalization(x, mm, mv, b, g, epsilon)
                batch_mean = tf.concat([moving_mean] * split_num, 0)
                batch_var = tf.concat([moving_var] * split_num, 0)
            else:
                # avoid the reshape if possible (when channel is the last dimension)
                xn = tf.nn.batch_normalization(
                    x, moving_mean, moving_var, beta, gamma, epsilon)
                batch_mean = tf.concat([moving_mean] * split_num, 0)
                batch_var = tf.concat([moving_var] * split_num, 0)

    # maintain EMA only on one GPU is OK, even in replicated mode.
    # because training time doesn't use EMA
    if ctx.is_main_training_tower:
        add_model_variable(moving_mean)
        add_model_variable(moving_var)
    if ctx.is_main_training_tower and use_local_stat:
        ret = update_bn_ema(xn, batch_mean[:n_out], batch_var[:n_out], moving_mean, moving_var, decay, internal_update)
    else:
        ret = tf.identity(xn, name='output')
    ret = tf.identity(xn, name='output')

    vh = ret.variables = VariableHolder(mean=moving_mean, variance=moving_var)
    if use_scale:
        vh.gamma = gamma
    if use_bias:
        vh.beta = beta
    assert batch_mean is not None, 'batch_mean outputs is None'
    return ret, batch_mean, batch_var
