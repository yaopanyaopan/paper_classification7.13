# -*-coding:utf-8 -*-
__author__ = '$'

import csv


def lingmingdu_5(all_predictions,y_test,m):
    A=0
    B=0
    C=0
    D=0
    A3=0
    A4=0
    B4=0
    for index,key in enumerate(all_predictions):
        if key==m:
            A4 +=1
            if y_test[index]==key:
                A +=1
            else:
                B +=1
        else:
            B4 +=1
    for index,key in enumerate(y_test):
        if key==m:
            A3 +=1
    if A3!=0:
        print(m,'灵敏度:',A / A3)
    if A4!=0:
        print(m,'特异度：',A / A4)
    if A3!=0 and A4!=0:
        return (A/A3,A/A4)
    else:
        return (0,0)

def lingmingdu_2(all_predictions,y_test):
    A=0
    B=0
    C=0
    D=0
    A3=0
    B3=0
    A4=0
    B4=0
    for index,key in enumerate(all_predictions):  # 0,3为０类，１２４为１类
        if key==0 or key==3:
            A4 +=1
            if y_test[index]==0 or y_test[index]==3:
                A +=1
            else:
                B +=1
        else:
            B4 +=1
            if y_test[index]==1 or y_test[index]==2 or y_test[index]==4:
                D +=1
            else:
                C +=1
    for index,key in enumerate(y_test):
        if key==0 or key==3:
            A3 +=1
        else:
            B3 +=1
    print('0,3/0类:','灵敏度:',A / A3)
    print('0,3/0类:','特异度：',A / A4)
    print('1,2,4/1类:', '灵敏度:', D / B3)
    print('1,2,4/1类:', '特异度：', D / B4)

    return (A/A3,A/A4,D/B3,D/B4)

