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
import lstm_model
from lstm_model import LSTM_Attention
import csv
import jisuan
import data_helpers


# Parameters
# ==================================================


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


x_text1,y = data_helpers.load_data_and_labels('test.csv','train.csv')     # 载入测试集样例　与　标签
y = np.argmax(y,1)

max_length = max([len(x) for x in x_text1])     # 单个样例最大长度
print('max-length',max_length)

x_text = []
for x in x_text1:
    x=' '.join(x)
    x_text.append(x)

vocab_path = os.path.join('lstm_runs/1531792671','vocab')
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)    # 载入词表

x = np.array(list(vocab_processor.transform(x_text)))     # 将输入词转换为相对应索引

# 从scores中取出前五 get label using probs
def get_label_using_probs(scores, top_number=5):
    index_list = np.argsort(scores)[-top_number:]
    index_list = index_list[::-1]
    return index_list


# =====================评 估=============================
print("Evaluating...\n")

checkpoint_file = tf.train.latest_checkpoint('lstm_runs/1531792671/checkpoints')
# print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x1 = graph.get_operation_by_name('input_x1').outputs[0]     # 获取输入的占位符
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        predictions = graph.get_operation_by_name('output/predictions').outputs[0]
        b_size = graph.get_operation_by_name('batch_size').outputs[0]
        # Generate batches

        batches = data_helpers.batch_iter(
            list(x),
            1,
            1 ,shuffle=False)


        all_predictions = []
        for x_batch in batches:

            # print(type(x_batch[0][0][0]))
            batch_predictions = sess.run(predictions,{input_x1:x_batch , dropout_keep_prob:1.0 ,b_size:1})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        correct_predictions = float(sum(all_predictions ==y))
        print('测试集样例总数:{}'.format(len(y)))
        print('Accuracy: {}'.format(correct_predictions/float(len(y))))

        with open('lstm_result/score.csv', 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(['5类问题'])
            print('5类问题：')

            L0, T0 = jisuan.lingmingdu_5(all_predictions, y, 0.0)  # 返回(灵敏度，特异度)
            L1, T1 = jisuan.lingmingdu_5(all_predictions, y, 1.0)
            L2, T2 = jisuan.lingmingdu_5(all_predictions, y, 2.0)
            L3, T3 = jisuan.lingmingdu_5(all_predictions, y, 3.0)
            L4, T4 = jisuan.lingmingdu_5(all_predictions, y, 4.0)

            writer.writerows([['第几类', '灵敏度', '特异度'],
                              [0, L0, T0],
                              [1, L1, T1],
                              [2, L2, T2],
                              [3, L3, T3],
                              [4, L4, T4]])
            writer.writerow(['2类问题'])
            print('2类问题：')
            L0, T0, L1, T1 = jisuan.lingmingdu_2(all_predictions, y)
            writer.writerows([['第几类', '灵敏度', '特异度'],
                              [0, L0, T0],
                              [1, L1, T1]])
            l0 = 0
            l1 = 0
            l2 = 0
            l3 = 0
            l4 = 0
            for cla in y:
                if cla == 0.0:
                    l0 += 1
                elif cla == 1.0:
                    l1 += 1
                elif cla == 2.0:
                    l2 += 1
                elif cla == 3.0:
                    l3 += 1
                else:
                    l4 += 1
            print('测试集各类数量:')
            print('0/1/2/3/4', l0, ',', l1, ',', l2, ',', l3, ',', l4)
            writer.writerows([['测试集各类数量'],
                              [l0, l1, l2, l3, l4]])
    # 保存 评价结果
    predictions_human_readable = np.column_stack((np.array(x), all_predictions))
    out_path = os.path.join('result', 'prediction.csv')
    print('保存 evaluation 到 {0}'.format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)









