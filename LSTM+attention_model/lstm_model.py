import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

from MyAttention import attention

class LSTM_Attention(object):


    def BiRNN(self, x, dropout, scope, batch_size, embedding_size, sequence_length, hidden_units):
        n_input = embedding_size
        n_steps = sequence_length
        n_hidden = hidden_units
        n_layers = 1


        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            print(tf.get_variable_scope().name)

            def lstm_fw_cell():
                fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                return tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
            lstm_fw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell() for _ in range(n_layers)],
                                                     state_is_tuple=True)
            # ** 4.初始状态
            initial_state_fw = lstm_fw_cell_m.zero_state(batch_size, tf.float32)

            # Backward direction cell
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            print(tf.get_variable_scope().name)
            # bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            # lstm_bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
            # lstm_bw_cell_m = rnn.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)
            def lstm_bw_cell():
                bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                return tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
            lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell() for _ in range(n_layers)], state_is_tuple=True)
            initial_state_bw = lstm_bw_cell_m.zero_state(batch_size, tf.float32)
        # Get lstm cell output

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            # outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            outputs, _= tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_m,
                lstm_bw_cell_m,
                x,
                initial_state_fw = initial_state_fw,
                initial_state_bw = initial_state_bw,
                dtype=tf.float32)
            # outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            #         except Exception: # Old TensorFlow version only returns outputs not states
            #             outputs = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,
            #                                             dtype=tf.float32)
        # return outputs[-1]
            outputs = tf.concat([outputs[0],outputs[1]],axis=2)
        return outputs


    def __init__(
            self, sequence_length, embedding_size, hidden_units, l2_reg_lambda, batch_size, attention_size, num_classes):
        # Placeholders for input, output and dropout
        initializer = tf.contrib.layers.xavier_initializer()

        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.b_size = tf.placeholder(tf.int32, [], name='batch_size')    #不固定batch_size

        self.W = tf.get_variable(initializer=initializer, shape=[20000, embedding_size], name='embedding')

        self.embeded_chars = tf.nn.embedding_lookup(self.W, self.input_x1)
        # self.embeded_chars_expand = tf.expand_dims(self.embeded_chars, -1)  # 在-1位置增加一个维度

        l2_loss = tf.constant(0.0, name="l2_loss")

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1 = self.BiRNN(self.embeded_chars, self.dropout_keep_prob, "side1", self.b_size, embedding_size, sequence_length, hidden_units)
            # self.out2 = self.BiRNN(self.input_x2, self.dropout_keep_prob, "side2", self.b_size, embedding_size, sequence_length, hidden_units)

            # Attention layer
            self.attention_output1, self.alphas1 = attention(self.out1, attention_size, return_alphas=True)

            w_projection = tf.get_variable("w", [hidden_units * 2, num_classes],initializer=initializer,dtype=tf.float32)
            b_projection = tf.get_variable("b", [num_classes], dtype=tf.float32)
            self.scores = tf.nn.xw_plus_b(self.attention_output1, w_projection, b_projection, name="scores")
            self.predictions = tf.argmax(self.scores,1,name='predictions')

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope('accuracy'):

            correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')





