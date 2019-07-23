# -*-coding:utf-8 -*-
__author__ = '$'

import numpy as np
import tensorflow as tf
import os
import time
import datetime
import matplotlib.pyplot as plt
from cnn import TextCNN
import data_helpers
from tensorflow.contrib import learn

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


FLAGS = tf.flags.FLAGS
# FLAGS._parse_args()
print('Loading data...')

x_text1,y = data_helpers.load_data_and_labels('train.csv','test.csv')     # 载入训练样例　与　标签

max_length = max([len(x) for x in x_text1])     # 单个样例最大长度
print('max-length:',max_length)

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length = max_length)     # 建立 vocabulary

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

# Training
print('开始训练...')

# for d in range(0,4):

with tf.device('/gpu:%s' % 3):

    with tf.name_scope('%s_%s' % ('tower', 3)):

      with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            # allow_soft_placement=True,
             log_device_placement=False,
        )
        # session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)  # 初始化 session
        with sess.as_default():
            cnn = TextCNN(
                sequence_length = x_train.shape[1],
                num_classes = 1,
                vocab_size = len(vocab_processor.vocabulary_),
                embedding_size = 100,
                filter_sizes = [2,3,4,5],
                num_filters = 256,
                l2_reg_lambda = 0.001
            )
            threashold = 0.25   #仅用于人工观察图
            # 定义 training 过程
            global_step = tf.Variable(0,name='global_step',trainable=False)
            optimizer = tf.train.AdamOptimizer(0.0005)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)

            train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)

            # 记录 gradient 和 sparsity (画图)
            # grad_summaries = []
            # for g,v in grads_and_vars:    # 梯度　与　对应变量
            #     if g is not None:
            #         grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name),g)
            #         sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name),tf.nn.zero_fraction(g))
            #         grad_summaries.append(sparsity_summary)
            #         grad_summaries.append(grad_hist_summary)
            # grad_summaries_merged = tf.summary.merge(grad_summaries)

            # 输出 model和 summary的路径
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir,'runs',timestamp))   # 存储模型的路径
            print('writing to {}\n'.format(out_dir))

            loss_summary = tf.summary.scalar('loss',cnn.loss)
            # acc_summary = tf.summary.scalar('accuracy',cnn.accuracy)

            # train summary (在图计算时，生成summary data)
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir,'summaries','train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            # 验证 summaries
            # dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir,'summaries','dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir , sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir,'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir,'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables())

            # 保存 vocabulary
            vocab_processor.save(os.path.join(out_dir,'vocab'))
            # 初始化所有 variables
            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch,writer=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob:0.9
                }
                _,step,summaries,loss,predictions = sess.run(
                    [train_op,global_step,train_summary_op,cnn.loss,cnn.predictions],feed_dict
                )

                acc_count = 0
                for index_y, pre in enumerate(predictions):
                    if pre > threashold and y_batch[index_y] == 1:
                        acc_count += 1
                    elif pre <= threashold and y_batch[index_y] == 0:
                        acc_count += 1

                acc = float(acc_count) / len(y_batch)
                print('step {} , loss {:g} , acc {} '.format(step,loss,acc))
                if writer:
                    writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch,max_acc,writer=None):
                # 在验证集上 评价模型
                feed_dict={
                    cnn.input_x:x_batch,
                    cnn.input_y:y_batch,
                    cnn.dropout_keep_prob:1.0
                }
                step,loss , predictions= sess.run(
                    [global_step,cnn.loss,cnn.predictions],feed_dict
                )
                acc_count = 0
                for index_y,pre  in enumerate(predictions):
                    if pre>threashold and y_batch[index_y]==1:
                        acc_count +=1
                    elif pre<=threashold and y_batch[index_y]==0:
                        acc_count +=1

                acc =  float(acc_count)/len(y_batch)
                print('step {} , loss {:g} , max_acc {:g} , acc {}'.format(step,loss,max_acc, acc))
                # if writer:
                #     writer.add_summary(summaries,step)

                if max_acc < acc:
                        max_acc = acc
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print('Saved model checkpoint to {}\n'.format(path))
                return max_acc,acc,loss


            batches = data_helpers.batch_iter(
                list(zip(x_train,y_train)),
                64,
                21
            )

            # 训练 Training loop
            step = []
            dev_acc = []
            dev_loss = []

            max_acc = 0
            for batch in batches:                  #  遍历每一批  数据
                x_batch,y_batch = zip(*batch)

                train_step(x_batch,y_batch)

                current_step = tf.train.global_step(sess,global_step)

                if current_step % 50==0:   # 每50步验证一次
                    print('\n Evaluation:')
                    max_acc ,d_acc , d_loss = dev_step(x_dev,y_dev,max_acc,writer=dev_summary_writer)

                    step.append(current_step)
                    dev_acc.append(d_acc)
                    dev_loss.append(d_loss)
                    print('')


        def to_picture(title, x_content, y_content, xlabel, ylabel, xlim, ylim,xticks,yticks, path):
            print("    - [Info] Plotting metrics into picture " + path)

            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['axes.unicode_minus'] = True

            plt.figure(figsize=(10, 5))
            plt.grid(linestyle="--")
            plt.xlim(xlim)
            plt.ylim(ylim)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.title(title, fontsize=14, fontweight='bold')
            plt.plot(x_content, y_content)
            plt.xticks(xticks)
            plt.yticks(yticks)
            plt.xlabel(xlabel, fontsize=13, fontweight='bold')
            plt.ylabel(ylabel, fontsize=13, fontweight='bold')
            plt.savefig(path, format='png')
            plt.clf()

        out_dir = os.path.join(out_dir)
        to_picture(title='dev-acc', x_content=step, y_content=dev_acc, xlabel='Epoch', ylabel='acc', xlim=(0, 1000),ylim=(0.5,1),xticks=np.linspace(0,1000,21,endpoint=True),yticks=np.linspace(0.5,1,11,endpoint=True),
                   path=out_dir+'/' + 'dev_acc.png')
        to_picture(title='dev_loss', x_content=step, y_content=dev_loss, xlabel='Epoch', ylabel='loss', xlim=(0, 1000),ylim=(0,0.1),xticks=np.linspace(0,1000,21,endpoint=True),yticks=np.linspace(0,0.1,11,endpoint=True),
                   path=out_dir +'/'+ 'dev_loss.png')



