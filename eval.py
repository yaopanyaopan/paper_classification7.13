# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from cnn import TextCNN
from tensorflow.contrib import learn
import csv
import jisuan

FLAGS = tf.flags.FLAGS

print('\n Parameters:')
for attr, value in sorted(FLAGS._flags().items()):
    print('{}={}'.format(attr.upper(),value))
print('')

x_raw1 , y_test = data_helpers.load_data_and_labels('test.csv','train.csv')
# y_test = np.argmax(y_test,1)
res = []
for i in y_test:
    for j in i:
        res.append(j)
y_test = res

x_raw = []
for x in x_raw1:
    x=' '.join(x)
    x_raw.append(x)

vocab_path = os.path.join('runs/1558784589','vocab')
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)       # 载入词向量
x_test = np.array(list(vocab_processor.transform(x_raw)))

#----------------------评 估----------------------------------------------------------
print('\nEvaluating...\n')

checkpoint_file = tf.train.latest_checkpoint('runs/1558784589/checkpoints')

graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 载入保存的 数据图 和 保存的变量
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess,checkpoint_file)

        input_x = graph.get_operation_by_name('input_x').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        predictions = graph.get_operation_by_name('output/predictions').outputs[0]

        batches = data_helpers.batch_iter(list(x_test),1,1,shuffle=False)

        all_predictions = []

        for x_test_batch in batches:

            batch_predictions = sess.run(
                predictions,{input_x:x_test_batch,dropout_keep_prob:1.0}
            )
            # print(batch_predictions)
            # all_predictions = np.concatenate([all_predictions,batch_predictions])
            all_predictions.append(batch_predictions[0])

if y_test is not None:

    # correct_predictions = float( sum(all_predictions == y_test))
    # print('测试样例总数：{}'.format(len(y_test)))
    # print('Accuracy: {:g}'.format(correct_predictions/float(len(y_test))))

    threashold = 0.09   # 用于得出多组 tp,fp,tn,fn
    acc_count = 0
    for index_y, pre in enumerate(all_predictions):

        if pre > threashold and y_test[index_y] == 1.0:
            acc_count += 1

        elif pre <= threashold and y_test[index_y] == 0:
            acc_count += 1

    acc = float(acc_count) / len(y_test)

    print(acc_count)
    print(len(y_test))
    print('acc:',acc)

    with open('result/score.csv', 'w') as fw:
        writer = csv.writer(fw)
        # writer.writerow(['５类问题'])
        # print('5类问题：')

        # L0,T0=jisuan.lingmingdu_5(all_predictions,y_test,0.0)    # 返回(灵敏度，特异度)
        # L1,T1=jisuan.lingmingdu_5(all_predictions,y_test,1.0)
        # L2,T2=jisuan.lingmingdu_5(all_predictions,y_test,2.0)
        # L3,T3=jisuan.lingmingdu_5(all_predictions,y_test,3.0)
        # L4,T4=jisuan.lingmingdu_5(all_predictions,y_test,4.0)
        #
        # writer.writerows([['第几类','灵敏度','特异度'],
        #                       [0, L0, T0],
        #                       [1, L1, T1],
        #                       [2, L2, T2],
        #                       [3, L3, T3],
        #                       [4, L4, T4]])
        # writer.writerow(['2类问题'])
        # print('2类问题：')
        # L0,T0,L1,T1 = jisuan.lingmingdu_2(all_predictions,y_test)
        # writer.writerows([['第几类','灵敏度','特异度'],
        #                   [0,L0,T0],
        #                   [1,L1,T1]])
        # l0=0
        # l1=0
        # l2=0
        # l3=0
        # l4=0
        # for cla in y_test:
        #     if cla==0.0:
        #         l0 +=1
        #     elif  cla==1.0:
        #         l1 +=1
        #     elif  cla==2.0:
        #         l2 +=1
        #     elif  cla==3.0:
        #         l3 +=1
        #     else:
        #         l4 +=1
        # print('测试集各类数量:')
        # print('0/1/2/3/4',l0,',',l1,',',l2,',',l3,',',l4)
        # writer.writerows([['测试集各类数量'],
        #                   [l0,l1,l2,l3,l4]])

        # 5星，非5星
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for index,i in enumerate(all_predictions):

                if y_test[index]==1 and i > threashold:
                    tp +=1
                elif y_test[index]==0 and i <= threashold:
                    tn +=1
                elif y_test[index]==0 and i > threashold:
                    fp +=1
                elif y_test[index]==1 and i <= threashold:
                    fn +=1

        print('tp',tp)
        print('fp',fp)
        print('tn',tn)
        print('fn',fn)

# 保存 评价结果
predictions_human_readable = np.column_stack((np.array(x_raw),y_test,all_predictions))
out_path = os.path.join('result','prediction.csv')
print('保存 evaluation 到 {0}'.format(out_path))
with open(out_path,'w') as f:
    csv.writer(f).writerows(predictions_human_readable)

with open('result/no_equal.csv','w') as fw:
    writer1= csv.writer(fw)
    writer1.writerow(['正文','真实值','预测值'])
    for i in range(len(y_test)):
        if y_test[i]!=all_predictions[i]:
            writer1.writerow([np.array(x_raw[i]) ,y_test[i] ,all_predictions[i]])



