# -*-coding:utf-8 -*-
__author__ = '$'
import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime

from lstm_model import LSTM_Attention

import data_helpers

# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 200, "Number of hidden units in softmax regression layer (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("sentence_words_num", 20, "The number of words in each sentence (default: 30)")
tf.flags.DEFINE_integer("attention_size", 30, "attention Size (default: 50)")
tf.flags.DEFINE_integer("num_classes", 5, "numble of classes")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files == None:
    print("Input Files List is empty. use --training_files argument.")
    exit()


x_text1,y = data_helpers.load_data_and_labels('train.csv','test.csv')     # 载入训练样例　与　标签

max_length = max([len(x) for x in x_text1])     # 单个样例最大长度
print('max-length',max_length)

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length= max_length)
x_text = []
for x in x_text1:
    x=' '.join(x)
    x_text.append(x)

x = np.array(list(vocab_processor.fit_transform(x_text)))     # 将输入词转换为相对应索引

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))    # 随机打乱
x_shuffled = x[shuffle_indices]
y_shuffled = np.array(y)[shuffle_indices]

dev_sample_index = 1* int(0.8 * float(len(y)))

x_train,x_dev = x_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]     # 划分训练集　验证集
y_train,y_dev = y_shuffled[:dev_sample_index],y_shuffled[dev_sample_index:]
# y_dev = np.array([[float(k)] for k in y_dev])

del x,y,x_shuffled,y_shuffled

print('Vocabulary 大小: {:d}'.format(len(vocab_processor.vocabulary_)))
print('Train / Dev split : {:d} / {:d}'.format(len(y_train),len(y_dev)))


# 从scores中取出前五 get label using probs
def get_label_using_probs(scores, top_number=5):
    index_list = np.argsort(scores)[-top_number:]
    index_list = index_list[::-1]
    return index_list


# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        Model = LSTM_Attention(
            sequence_length = x_train.shape[1],
            embedding_size = FLAGS.embedding_dim,
            hidden_units = FLAGS.hidden_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size,
            attention_size=FLAGS.attention_size,
            num_classes=FLAGS.num_classes)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        print("initialized siameseModel object")

    grads_and_vars = optimizer.compute_gradients(Model.cost)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir,'LSTM+attention_model', "lstm_runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    loss_summary = tf.summary.scalar("loss", Model.cost)
    # acc_summary = tf.summary.scalar("accuracy", Model.accuracy)

    train_summary_op = tf.summary.merge([loss_summary,  grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))

    sess.run(tf.global_variables_initializer())
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(x1_batch, y_batch,writer=None):

        # print(x1_batch)
        feed_dict = {
            Model.input_x1: x1_batch,
            Model.input_y: y_batch,
            Model.dropout_keep_prob: FLAGS.dropout_keep_prob,
            Model.b_size: len(y_batch)
        }

        _, step, loss, scores,acc = sess.run(
            [tr_op_set, global_step, Model.cost, Model.scores,Model.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()


        print("TRAIN  step: %i , loss is: %f ,  acc is: %f" %
              (step, loss, acc))

        summary_op_out = sess.run(train_summary_op, feed_dict=feed_dict)
        train_summary_writer.add_summary(summary_op_out, step)
        # print (y_batch, dist, d)


    def dev_step(x1_batch, y_batch,max_acc,writer=None):

        feed_dict = {
            Model.input_x1: x1_batch,
            Model.input_y: y_batch,
            Model.dropout_keep_prob: 1.0,
            Model.b_size: len(y_batch)
        }

        step, loss, scores,acc = sess.run(
            [global_step, Model.cost, Model.scores,Model.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()

        # print("DEV {}: step {}, loss {:g}, f1 {:g}".format(time_str, step, loss, f1))
        print("DEV  step: %i , loss is: %f \n" %
              (step, loss))
        print('This step acc:{} , max_acc:{}'.format(acc, max_acc))
        if acc > max_acc:
            max_acc = acc
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print('Saved model to:{}\n'.format(path))
            print('------------------------')
        return max_acc


    # Generate batches
    # print(x_train)
    # print(list(zip(x_train,y_train)))
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)),
        64,
        200
    )
    max_acc = 0
    # print('batches')
    # print(batches)

    for batch in batches:
        # print(batch)
        x_batch, y_batch = zip(*batch)

        train_step(x_batch, y_batch)

        current_step = tf.train.global_step(sess, global_step)

        if current_step % 100 == 0:
            print('-----------------------')
            print('\nEvaluation:')
            max_acc = dev_step(x_dev, y_dev, max_acc,writer=dev_summary_writer)





