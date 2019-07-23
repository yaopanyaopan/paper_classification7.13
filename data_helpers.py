# -*-coding:utf-8 -*-
__author__ = '$'


import numpy as np
import re
import itertools
from collections import Counter
import os
import csv
import jieba
import random
import collections
import gensim
from gensim import *


def count_tf():
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    with open('data.csv','r') as fr:
        reader = csv.reader(fr)

        for i in reader:
            # print(i)
            if i[2].strip()=='1':
                data1.append([i[0],i[1]])
            elif i[2].strip()=='2':
                data2.append([i[0],i[1]])
            elif i[2].strip()=='3':
                data3.append([i[0],i[1]])
            elif i[2].strip()=='4':
                data4.append([i[0],i[1]])
            else:
                data5.append([i[0],i[1]])

        for index,k in enumerate([data1,data2,data3,data4,data5]):
            dict1 = {}
            for i in k:
                s1 = clear_data(i[0])
                s2 = clear_data(i[1])

                s1[0:0] = s2

                # print(s1)
                for j in s1:
                    if j not in dict1.keys():

                        dict1[j] = 1
                    else:

                        dict1[j] +=1

            # print(dict1.keys())
            path = 'tf_count/' + str(index) +'_tf.csv'
            print(index)
            with open(path,'w') as fw:

                writer = csv.writer(fw)
                for k,v in dict1.items():
                    # print(k,v)
                    writer.writerow([k,v])




def data_process():

   data_list = os.listdir('data')
   print(data_list)
   data_list.sort()
   print(data_list)

   with open('data.csv','w',newline='') as fw:
       writer = csv.writer(fw)
       for i in data_list:
           label = i[0]
           print(label)
           i ='data/'+i
           with open(i,'r') as fr:
               a=[]
               for line in fr:
                   # print(line)
                   if '%T' in line:
                       a.append(line.strip().split('%T')[1])
                       # print(line)
                   elif '%X' in line:
                       # print(line)
                       a.append(line.strip().split('%X')[1])
                   elif len(line.strip())==0:
                       if len(a)>1:
                           a.append(label)
                           writer.writerow(a)
                           a = []
                       # print(line)
                   else:
                       pass


def DATA_train_test():

    with open('train.csv','w',newline='') as fw1:
        writer1 = csv.writer(fw1)

        with open('test.csv','w',newline='') as fw2:
            writer2 = csv.writer(fw2)
            D_train=[]
            D_test=[]
            D1 = []
            D2 = []
            D3 = []
            D4 = []
            D5 = []
            with open('data.csv','r') as fr:
                reader = csv.reader(fr)
                for i in reader:
                    if i[2]=='1':
                        print(i)
                        D1.append(i)
                    if i[2]=='2':
                        print(i)
                        D2.append(i)
                    if i[2]=='3':
                        print(i)
                        D3.append(i)
                    if i[2]=='4':
                        print(i)
                        D4.append(i)
                    if i[2]=='5':
                        print(i)
                        D5.append(i)

                for index,key in enumerate(D1):
                    if index < 0.8*len(D1):
                        D_train.append([key[0],key[1],key[2]])

                    else:
                        D_test.append([key[0],key[1],key[2]])
                for index,key in enumerate(D2):
                    if index < 0.8*len(D2):
                        D_train.append([key[0], key[1], key[2]])
                    else:
                        D_test.append([key[0], key[1], key[2]])
                for index,key in enumerate(D3):
                    if index < 0.8*len(D3):
                        D_train.append([key[0], key[1], key[2]])
                    else:
                        D_test.append([key[0], key[1], key[2]])
                for index,key in enumerate(D4):
                    if index < 0.8*len(D4):
                        D_train.append([key[0], key[1], key[2]])
                    else:
                        D_test.append([key[0], key[1], key[2]])
                for index,key in enumerate(D5):
                    if index < 0.8*len(D5):
                        D_train.append([key[0], key[1], key[2]])
                    else:
                        D_test.append([key[0], key[1], key[2]])
                writer1.writerows(D_train)
                writer2.writerows(D_test)

def clear_data(str):

   ss = jieba.cut(str.strip())
   stop_words = []
   with open('stop_words.txt') as f:
       for i in f:
           # print(i)
           stop_words.append(i.strip())
   string = []
   for i in ss:
       if i in stop_words:
           pass
       else:

           if re.sub(r'[^A-Za-z0-9(),!?\'\`]', "", i):
               pass
           else:
               if len(i.strip())==0:
                   pass
               else:
                   string.append(i)
   # print(string)
   return string



def load_data_and_labels(train_data_file , test_data_file):

    with open(train_data_file,'r') as fr:
        reader = csv.reader(fr)
        negative_data_examples= open(test_data_file,'r').readlines()

        # 分词
        train_data_example = []
        train_data_labels = []
        data = []
        for i in reader:
            data.append(i)

        data = np.random.permutation(data)
        print(type(data))


        count_5star = 0
        for i in data:
            train_data_example.append(clear_data(i[1]))
            # if int(i[2].strip())==1:
            #     train_data_labels.append([1,0,0,0,0])
            # elif int(i[2].strip())==2:
            #     train_data_labels.append([0 ,1, 0, 0, 0])
            # elif int(i[2].strip())==3:
            #     train_data_labels.append([0 ,0, 1, 0, 0])
            # elif int(i[2].strip())==4:
            #     train_data_labels.append([0 ,0, 0, 1, 0])
            # else :
            #     train_data_labels.append([0 ,0, 0, 0, 1])
            if int(i[2].strip())==5:
                train_data_labels.append([1])
                count_5star +=1
            else:
                train_data_labels.append([0])

        print('总数：',len(data))
        print('5star : ',count_5star)
        return [train_data_example ,train_data_labels]


def batch_iter(data , batch_size , num_epochs , shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int( (len(data)-1)/batch_size) + 1    # 每一次周期 有多少批的数据

    for epoch in range(num_epochs):         # 在一个 epoch ，训练所有数据
        print('epoch : ', epoch)
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        for batch_num in range(num_batches_per_epoch):      # 遍历 每一批 数据
            start_index = batch_num * batch_size
            end_index = min((batch_num +1)* batch_size,data_size)

            yield shuffle_data[start_index:end_index]









if __name__=='__main__':

     # data_process()

     # DATA_train_test()
    # load_data_and_labels('train.csv','test.csv')

     count_tf()
















