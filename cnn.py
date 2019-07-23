# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np


class TextCNN(object):


    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size
                 ,filter_sizes,num_filters,l2_reg_lambda = 0.0):

        initializer = tf.contrib.layers.xavier_initializer()
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name='input_x')    # 列是sequence_length,行不定长
        self.input_y = tf.placeholder(tf.float32,[None,num_classes],name= 'input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)  # 初始化 loss

        # embedding 层
        with tf.device('/cpu:0') , tf.name_scope('embedding'):
            # W 代表词 embedding
            self.W = tf.get_variable(initializer=initializer,shape=[20000,embedding_size],name='WW')     # 建立词表
            self.embeded_chars = tf.nn.embedding_lookup(self.W , self.input_x)     # 在参数 W中查找索引为self_input的表示
            self.embeded_chars_expand = tf.expand_dims(self.embeded_chars, -1)       # 在-1位置增加一个维度

        # conv + maxpool
        pooled_outputs = []   # 要拼接 pooling后值为一个向量
        for i,filter_size in enumerate(filter_sizes):   # 遍历每一个filter
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                filter_shape = [filter_size ,embedding_size ,1 ,num_filters] # 步长为1
                # W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')  # 正太分布
                name ='W%s' % i
                W = tf.get_variable(initializer=initializer,shape=filter_shape,name=name)
                b = tf.Variable(tf.constant(0.0,shape=[num_filters]),name='b')

                conv = tf.nn.conv2d(self.embeded_chars_expand,
                                    W,
                                    strides = [1,1,1,1],
                                    padding = 'VALID',
                                    name = 'conv'
                                    )   #batch_size_stride、height_stride、width_stride、channels_stride
                # 应用 非线性函数
                h = tf.nn.relu(tf.nn.bias_add(conv,b), name='relu')

                # maxpooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1,sequence_length - filter_size + 1,1,1],   # 对整个卷积后的向量做池化
                    strides=[1,1,1,1],
                    padding='VALID',
                    name = 'pool'
                )
                pooled_outputs.append(pooled)   # 拼接每一个 filter得到的结果


        # 拼接所有 ， 池化后的特征
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat( pooled_outputs , 3)
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

        # dropout  ( 防止过拟合 )
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat , self.dropout_keep_prob)

        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape = [num_filters_total , num_classes],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable( tf.constant(0.0,shape = [num_classes]),name='b')

            l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop,W,b,name='scores')
            # self.predictions = tf.arg_max(self.scores,1,name='predictions')

            self.predictions = tf.sigmoid(self.scores,name='predictions')

            # 计算 交叉熵
            with tf.name_scope('loss'):

                # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)

                losses = -self.input_y*tf.log(tf.clip_by_value(self.predictions,1e-8,tf.reduce_max(self.predictions))) -(1-self.input_y)*tf.log(1-tf.clip_by_value(self.predictions,1e-8,tf.reduce_max(self.predictions)))
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda *l2_loss

            # 准确率
            # with tf.name_scope('accuracy'):
            #
            #     correct_predictions = tf.equal(self.predictions,tf.arg_max(self.input_y,1))
            #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')





















