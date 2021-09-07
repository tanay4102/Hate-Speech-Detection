# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:52:01 2020

@author: tanay
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
import spacy
import fasttext
#RANDOM FOREST TASK
print("...Processing Data Set for TFIDF")
f=open("train-preprocess.tsv",'r')
#Initialising BagofWords 
uniquewords=[]
corpus=[]
oi=0
for x in f:
    if oi!=0:
        y=x.split("\t")
        corpus.append(y[1])
        z=y[1].split(" ")
        z=np.unique(z)
        uniquewords=set(z).union(set(uniquewords))
    oi=oi+1
#Removing Required words (more than 80% less than 5)
stopwords=[]
docfreq=dict.fromkeys(uniquewords, 0)
total=0
oi=0
f=open("train-preprocess.tsv",'r')
for x in f:
    if oi!=0:
        total=total+1
        y=x.split("\t")
        z=y[1].split(" ")
        z=np.unique(z)
        for l in z:
            docfreq[l]=docfreq[l]+1
    oi=oi+1
for x in docfreq:
    if docfreq[x]>0.8*total:
        stopwords.append(x)
    if docfreq[x]<5:
        stopwords.append(x)
uniquewords=uniquewords.difference(stopwords) 
#Initialising TFIDF Vector  for the data set
vectorizer = TfidfVectorizer(vocabulary=uniquewords)
X = vectorizer.fit_transform(corpus)
print("Processing Complete for TFIDF")
#getting binary ouputs for hate speech
oi=0
output=[]
f=open("train-preprocess.tsv",'r')
for x in f:
    if oi!=0:
        y=x.split("\t")
        output.append(int(y[2]))
    oi=oi+1
output=np.array(output)
X_train, X_test, y_train, y_test = train_test_split(X, output, test_size=0.15)
# Random Forest Classifier for the above data
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
#Checking Accuracy
y_pred=clf.predict(X_test)
print("F1 score of Random forest on 85-15 split:",metrics.f1_score(y_test, y_pred,average='macro'))
#Predicting Data for the Test Sets
f=open("test-preprocess.tsv",'r')
#Initialising new corpus for test set
corpus_test=[]
oi=0
for x in f:
    if oi!=0:
        y=x.split("\t")
        y[1]=y[1].rstrip()
        corpus_test.append(y[1])
    oi=oi+1 
#Initialising TFIDF Vector  for the test set
vectorizer_test = TfidfVectorizer(vocabulary=uniquewords)
XTEST = vectorizer_test.fit_transform(corpus_test)
YTEST=clf.predict(XTEST)
g=open("RF.csv",'w')
f=open("test-preprocess.tsv",'r')
oi=0
for x in f :
    y=x.split("\t")
    if oi==0:
        g.write(y[0])
        g.write(",")
        g.write("hateful")
        g.write("\n")
    else:
        g.write(y[0])
        g.write(",")
        g.write(str(YTEST[oi-1]))
        g.write("\n")
    oi=oi+1
#END OF RANDOM FOREST TASK
#START OF SVM FEATURE VECTORS
print("...Processing data for SVM (will take around 7 minutes)")
#function making feature vectors
def get_mean_vector(nlp, doc):
    # remove out-of-vocabulary words
    zero=[0]*300
    zero=np.array(zero)
    tokens=nlp(doc)
    if len(tokens) >= 1:
        data=[]
        for word in tokens:
            data.append(word.vector)
        data=np.array(data)
        data=np.average(data,axis=0)
        return data
    else:
        return zero
nlp = spacy.load("en_core_web_md")
#generating Bag of Words and embedding vectors
f=open("train-preprocess.tsv",'r')
featurevectors=[]
output=[]
oi=0
for x in f:
    if oi!=0:
        if (oi%5000==0):
            print("Processing done till Tweet ",oi)
        y=x.split("\t")
        tokens=nlp(y[1])
        vec = get_mean_vector(nlp, y[1])
        if len(vec) > 0:
            output.append(int(y[2]))
            featurevectors.append(vec)
    oi=oi+1
featurevectors=np.array(featurevectors)
output=np.array(output)
print("Processing done for SVM")
#print(featurevectors.shape())
#Validation Set
X_train, X_test, y_train, y_test = train_test_split(featurevectors, output, test_size=0.15)
clf=svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("F1 score for SVM on a 85-15 split:",metrics.f1_score(y_test, y_pred,average='macro'))
corpus_test=[]
oi=0
f=open("test-preprocess.tsv",'r')
for x in f:
    if oi!=0:
        y=x.split("\t")
        y[1]=y[1].rstrip()
        temp=get_mean_vector(nlp,y[1])
        if len(temp)>0:
            corpus_test.append(temp)
    oi=oi+1 
corpus_test=np.array(corpus_test)
YTEST=clf.predict(corpus_test)
g=open("SVM.csv",'w')
f=open("test-preprocess.tsv",'r')
oi=0
for x in f :
    y=x.split("\t")
    if oi==0:
        g.write(y[0])
        g.write(",")
        g.write("hateful")
        g.write("\n")
    else:
        g.write(y[0])
        g.write(",")
        g.write(str(YTEST[oi-1]))
        g.write("\n")
    oi=oi+1
#END OF SVM 
#FASTTEXT
print("...Processing data for fasttext")
f=open("train-preprocess.tsv",'r')
g=open("fasttexttrain.txt",'w')
oi=0
pred=[]
out=[]
for x in f:
    y=x.split("\t")
    if oi!=0 and oi<int(0.85*16113):
        g.write(y[1])
        g.write(" ")
        s="__label__"+y[2]
        g.write(s)
        g.write("\n")
    elif oi>=int(0.85*16113) :
                pred.append(y[1])
                out.append(int(y[2]))
        
    oi=oi+1
out=np.array(out)
print("Processing done for fasttext")
model = fasttext.train_supervised('fasttexttrain.txt')
y=[]
#print(len(pred))
s=str(model.predict(pred)[0])
#s="".join(s)
for x in s:
    if x=='0' or x=='1' :
        y.append(int(x))
print("F1 score of fasttext on 85-15 split:",metrics.f1_score(out, y,average='macro'))
#os.remove('fasttexttrain.txt')
f=open("test-preprocess.tsv",'r')
xtest=[]
idno=[]
oi=0
for x in f:
    y=x.split("\t")
    if oi!=0:
        y[1]=y[1].rstrip()
        xtest.append(y[1])
        idno.append((y[0]))
    oi=oi+1
sfinal=str(model.predict(xtest)[0])
print(len(idno))
#print(sfinal)
g=open('FT.csv','w')
g.write("id")
g.write(",")
g.write("hateful")
g.write("\n")
oi=0
for x in sfinal:
    if x=='0' or x=='1' :
        #print(x)
        g.write((idno[oi]))
        g.write(",")
        g.write(x)
        g.write("\n")
        oi=oi+1
g.close()
print(oi)
print("End of Task 1")
##END OF TASK 1
    