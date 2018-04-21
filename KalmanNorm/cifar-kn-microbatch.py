#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-resnet.py
# Author: Guangrun Wang <https://wanggrun.github.io/>

import argparse
import os


from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow import dataset
from kalman_norm import KalmanNorm as KN

import tensorflow as tf
import numpy as np

"""
CIFAR10 ResNet example. See:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
This implementation uses the variants proposed in:
Identity Mappings in Deep Residual Networks, arxiv:1603.05027
"""

BATCH_SIZE = 128
NUM_UNITS = None
split_num = 64
print('split number is : {}'.format(split_num))


class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()
        #image = tf.transpose(image, [0, 3, 1, 2])

        def state_input(statistic_shape, m1, v1):
            statistic_ini = tf.random_normal(statistic_shape,mean=0,stddev=1.0,dtype=tf.float32)
            # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
            m1 = tf.expand_dims(m1, 0)
            m1 = tf.expand_dims(m1, 0)
            m1 = tf.expand_dims(m1, 0)
            v1 = tf.expand_dims(v1, 0)
            v1 = tf.expand_dims(v1, 0)
            v1 = tf.expand_dims(v1, 0)
            statistic_input = statistic_ini * tf.sqrt(v1)
            statistic_input = statistic_input + m1
            statistic_input = tf.nn.relu(statistic_input)
            return statistic_input

        def residual(name, l, m, v, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name):
                if first: 
                    b1 = l 
                    m1 = m
                    v1 = v
                else:
                    b1, m1, v1 = KN('kn', l, m, v, split_num = split_num, p_rate = 0.9)
                    b1 = tf.nn.relu(b1)

                b1_shape = b1.get_shape().as_list()
                statistic_shape = [1] + b1_shape[1:3] + [b1_shape[-1]*split_num]
                # print (statistic_shape, b1_shape)           
                statistic_input = state_input(statistic_shape, m1, v1)
                statistic_input = tf.concat(tf.split(statistic_input, split_num, 3), 0)                            
                with tf.variable_scope("conv1_kn") as scope:
                    c1 = Conv2D('conv1', b1, out_channel, strides=stride1)
                    scope.reuse_variables()
                    statistic_input = Conv2D('conv1', statistic_input, out_channel, strides=stride1)
                    statistic_input = tf.concat(tf.split(statistic_input, split_num, 0), -1)
                    axis = [0, 1, 2]
                    m1, v1 = tf.nn.moments(statistic_input, axis) # C*split_num
                c1, m2, v2 = KN('kn1', c1, m1, v1, split_num = split_num, p_rate = 0.9)

                c1 = tf.nn.relu(c1)
                c1_shape = c1.get_shape().as_list()
                statistic_shape = [1] + c1_shape[1:3] + [c1_shape[-1]*split_num]
                statistic_input = state_input(statistic_shape, m2, v2)
                statistic_input = tf.concat(tf.split(statistic_input, split_num, 3), 0)
                with tf.variable_scope("conv2_kn") as scope:
                    c2 = Conv2D('conv2', c1, out_channel)
                    scope.reuse_variables()
                    statistic_input = Conv2D('conv2', statistic_input, out_channel)
                    statistic_input = tf.concat(tf.split(statistic_input, split_num, 0), -1)
                    axis = [0, 1, 2]
                    m3, v3 = tf.nn.moments(statistic_input, axis) # C*split_num

                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [0, 0], [0, 0], [in_channel // 2, in_channel // 2]])
                    m = tf.pad(m, [[in_channel*split_num // 2, in_channel*split_num // 2]])
                    v = tf.pad(v, [[in_channel*split_num // 2, in_channel*split_num // 2]])

                l = c2 + l
                return l, m+m3, v+v3

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_last'), \
                argscope(Conv2D, use_bias=False, kernel_size=3,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):

            l = Conv2D('conv0', image, 16)
            l, m, v = KN('kn0', l, None, None, split_num = split_num, p_rate = 0.9)
            print v.get_shape().as_list()
            l = tf.nn.relu(l)
            l, m, v = residual('res1.0', l, m, v, first=True)
            for k in range(1, self.n):
                l, m, v = residual('res1.{}'.format(k), l, m, v)
            # 32,c=16

            l, m, v = residual('res2.0', l, m, v, increase_dim=True)
            for k in range(1, self.n):
                l, m, v = residual('res2.{}'.format(k), l, m, v)
            # 16,c=32

            l, m, v = residual('res3.0', l, m, v, increase_dim=True)
            for k in range(1, self.n):
                l, m, v = residual('res3.' + str(k), l, m, v)
            l, _, _ = KN('knlast', l, m, v, split_num = split_num, p_rate = 0.9)
            l = tf.nn.relu(l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

        logits = FullyConnected('linear', l, 10)
        tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=5)
    parser.add_argument('--load', help='load model')

    parser.add_argument('--log_dir', type=str, default='')
    args = parser.parse_args()
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # log_foder = ''
    log_foder = '/data1/wangguangrun/train_log/cifar10-resnet-%s' % (args.log_dir)
    logger.set_logger_dir(os.path.join(log_foder))


    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ],
        max_epoch=400,
        session_init=SaverRestore(args.load) if args.load else None
    )
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
