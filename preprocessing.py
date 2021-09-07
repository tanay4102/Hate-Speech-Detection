# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:34:16 2020

@author: tanay
"""
punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
f=open("train.tsv",'r')
g=open("train-preprocess.tsv",'w')
for x in f:
    y=x.split('\t')
    g.write(y[0])
    for ele in y[1]:  
        if ele in punc:  
            y[1] = y[1].replace(ele, "")
    g.write('\t')
    g.write(y[1])
    g.write('\t')
    g.write(y[2])
    print(y[1])
    
