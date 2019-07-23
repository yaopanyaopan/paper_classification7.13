import tensorflow as tf
import numpy as np

def attention(inputs, attention_size, time_major=False, return_alphas=False):

    # if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        # inputs = tf.concat(inputs, 2)
    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
    # inputs_shape = inputs.shape
    # sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    # hidden_size = inputs_shape[2].value  # hidden size of the RNN layer
    #将[batch_size, max_time, cell_fw.output_size]转换为[max_time, batch_size, cell_fw.output_size]
    # print(inputs)
    # inputs = tf.transpose(inputs, [0, 2, 1])
    # print(inputs)
    sequence_length = inputs.shape[1].value # the length of sequences processed in the antecedent RNN layer
    # print(sequence_length)
    hidden_size = inputs.shape[2].value # hidden size of the RNN layer
    # print(hidden_size)
    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    # print(alphas)
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
