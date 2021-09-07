# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:18:13 2020

@author: tanay
"""
import os
#preprocessingfortasktwo
save_path_hate="C:/Users/tanay/Desktop/ECE 3rd year/5 sem/Social Computing/assignement3/twt_sentoken/hate"
save_path_no_hate="C:/Users/tanay/Desktop/ECE 3rd year/5 sem/Social Computing/assignement3/twt_sentoken/nohate"
save_path_test="C:/Users/tanay/Desktop/ECE 3rd year/5 sem/Social Computing/assignement3/twt_sentoken/test"
f=open("train.tsv",'r')
oi=0
hate=0
nh=0
for x in f:
    y=x.split('\t')
    if oi!=0:
        if int(y[2])==0:
            if nh<9472 :
                name_of_file = "train"+y[0]
            else:
                name_of_file = "test"+y[0]
            completeName = os.path.join(save_path_no_hate, name_of_file+".txt")
            g=open(completeName,'w')
            g.write(y[1])
            nh=nh+1
            g.close()
        if int(y[2])==1:
            if hate<5029 :
                name_of_file = "train"+y[0]
            else:
                name_of_file = "test"+y[0]
            completeName = os.path.join(save_path_hate, name_of_file+".txt")
            g=open(completeName,'w')
            g.write(y[1])
            hate=hate+1
            g.close()
    oi=oi+1
print(nh,hate)
f=open("test.tsv",'r')
oi=0 
nhy=0
for x in f:
    y=x.split('\t')
    if oi!=0:
        name_of_file = "test"+y[0]
        completeName = os.path.join(save_path_test, name_of_file+".txt")
        g=open(completeName,'w')
        g.write(y[1])
        nhy=nhy+1
        g.close()
       
    oi=oi+1   
print(nhy)
