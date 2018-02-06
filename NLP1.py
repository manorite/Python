# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:04:11 2018

@author: Nitin PC
"""

import pandas as pd
import numpy as np 
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
# to download all the packages
#nltk.download()

url = 'C:\\Users\\Nitin PC\\Downloads\\quora_duplicate_questions.tsv'

data = pd.read_csv(url,sep='\t')
# understanding the data
data.head(5)
data.shape
data_colnames = data.columns
range_data = data.index
Y = data.values[:,5]
unique, count = np.unique(Y, return_counts= True)
no_info = count[0]/(count[0]+count[1])* 100
 
# feature Extraction 
# basic features 

data['length_q1'] = data.question1.apply(lambda x : len(str(x)))
data['length_q2'] = data.question2.apply(lambda x : len(str(x)))
data['ratio_q1_q2'] = data['length_q1'] / data['length_q2']
data['len_char_q1'] = data.question1.apply(lambda x : len(''.join(str(x).replace(' ',''))))
data['len_char_q2'] = data.question2.apply(lambda x : len(''.join(str(x).replace(' ',''))))
data['num_words_q1'] = data.question1.apply(lambda x : len(str(x).split(' ')))
data['num_words_q2'] = data.question2.apply(lambda x : len(str(x).split(' ')))
data['common_words'] = data.apply(lambda x :
                                    len(set(str(x['question1']).lower().split(' ')).
                                    intersection(set(str(x['question1']).lower().split(' ')))), axis=1)
# install fuzzywuzzy

# 2nd set of features     
from fuzzywuzzy import fuzz

data['qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['ratio'] = data.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
data['token_sort_ratio']= data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['token_set_ratio']= data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

data.head(3)

data.to_csv('C:\\Users\\Nitin PC\\Downloads\\preprocessed.tsv',sep='\t', encoding = 'utf-8')

data = pd.read_csv('C:\\Users\\Nitin PC\\Downloads\\preprocessed.tsv',sep='\t')
# more cleaning 
# stop words 

stop_words = stopwords.words('english')

def removing_stop_words(s1):
    s1 = str(s1).lower().split()
    s1 = [word for word in s1 if word not in stop_words]
    return(s1)

from nltk.stem.porter import *
ps = PorterStemmer()

def stemming(s1):
    s1 = str(s1).lower().split()
    s1 = [ps.stem(word) for word in s1 ]
    return(s1)

#import gensim as ge
#from nltk import word_tokenize   
#sent2vec_model = ge.models.KeyedVectors.load_word2vec_format('C:\\Users\\Nitin PC\\Downloads\\GoogleNews-vectors-negative300.bin.gz', binary=True)

from gensim.models import Word2Vec
from nltk.corpus import brown 

sentences = brown.sents()
word2vec_model = Word2Vec(sentences,min_count=1)
#print(word2vec_model.most_similar_cosmul('human'))
#print(word2vec_model['india'].sum())
#print(word2vec_model)

def sen2vect(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = filter(lambda x : x in word2vec_model.wv.vocab,words)
    # words = removing_stop_words(w) for w in words
    # create a empty vector 
    S_w = []
    for w in words:
        if w not in stop_words:
            S_w.append(word2vec_model[w])
        
    S_w = np.array(S_w)
    vector_sum = S_w.sum(axis=0)
    return (vector_sum/np.sqrt((vector_sum**2).sum())).mean()

# additional Features
    
data['sen2vec_q1'] = data.question1.apply(lambda x : sen2vect(x))
data['sen2vec_q2'] = data.question2.apply(lambda x : sen2vect(x))
# word vector distance 
from scipy.spatial.distance import cosine, cityblock, euclidean, jaccard
data['cosine'] = cosine(np.nan_to_num(data['sen2vec_q1']),np.nan_to_num(data['sen2vec_q2']))
data['cityblock'] = cityblock(np.nan_to_num(data['sen2vec_q1']),np.nan_to_num(data['sen2vec_q2']))
data['euclidean'] = euclidean(np.nan_to_num(data['sen2vec_q1']),np.nan_to_num(data['sen2vec_q2']))
data['jaccard'] = jaccard(np.nan_to_num(data['sen2vec_q1']),np.nan_to_num(data['sen2vec_q2']))

data.head(3)

data = data.drop(['Unnamed: 0','id', 'qid1', 'qid2', 'question1', 'question2','fuzz_qratio'], axis=1)
Y = data['is_duplicate']
x = data.drop(['is_duplicate'],axis=1)
x.isnull().sum()
x = x.fillna(0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

from scipy.stats import skew
skewness = skew(x)


import matplotlib.pyplot as plt 
plt.hist(x[3],log = True)

#from sklearn.preprocessing import Imputer
#x = Imputer(x,missing_values='NULL',strategy=0.0,axis=0)
x = pd.DataFrame(x)

x_train, x_test , y_train , y_test = train_test_split(x,Y, test_size = 0.30,stratify=Y)
# create validation set to for model tunning
x_val , x_test , y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5)

x_train.head(3)
x.corr() > 0.95
#import matplotlib.pyplot as plt

# creating a simple text 


# models machine learning 
# logistic regression @ 65% accuracy

from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model_log_reg = LogisticRegression()
model_log_reg.fit(x_train,y_train)
y_pred = model_log_reg.predict(x_val)
confusion_mat = confusion_matrix(y_val,y_pred)
print(model_log_reg.coef_)
print(accuracy_score(y_val,y_pred))
print(classification_report(y_val,y_pred))

# random forest model 
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
 
model_random_forest = RandomForestClassifier(n_jobs=2, max_features = 8,n_estimators = 100) # best @8 features 
model_random_forest.fit(x_train,y_train)
y_pred = model_random_forest.predict(x_val)
confusion_mat = confusion_matrix(y_val,y_pred)
roc = roc_auc_score(y_val,y_pred)
print(accuracy_score(y_val,y_pred))
print(classification_report(y_val,y_pred))
 
# naive bayes 

from sklearn.naive_bayes import MultinomialNB

model_NB = MultinomialNB()
model_NB.fit(x_train, y_train)
y_pred = model_NB.predict(x_val)
confusion_mat = confusion_matrix(y_val, y_pred)
print(accuracy_score(y_val,y_pred))
print(classification_report(y_val,y_pred))

# 
from sklearn.linear_model import SGDClassifier
model_SVM = SGDClassifier()
model_SVM.fit(x_train, y_train)
y_pred = model_SVM.predict(x_test)
confusion_mat = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

from sklearn import svm
kernel = ['linear','rbf','sigmoid']
for k in kernel:
    model_SVM =  svm.SVC(kernel='linear')
    model_SVM.fit(x_train, y_train)
    y_pred = model_SVM.predict(x_test)
    confusion_mat = confusion_matrix(y_test, y_pred)
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))

# neural Network @ 0.723925267549
from sklearn.neural_network import MLPClassifier
model_NN = MLPClassifier(activation ='relu',solver= 'lbfgs') 
model_NN.fit(x_train,y_train)
y_pred = model_NN.predict(x_val)
confusion_mat = confusion_matrix(y_val,y_pred)
roc = roc_auc_score(y_val,y_pred)
print(accuracy_score(y_val,y_pred))
print(classification_report(y_val,y_pred))


# final Model
# random Forest with 8 featues and 100 trees 

x_train = pd.DataFrame.append(x_train,x_val)
y_train = pd.Series.append(y_train,y_val)


final_model_random_forest = RandomForestClassifier(n_jobs=2, max_features = 8,n_estimators = 100) # best @8 features 
final_model_random_forest.fit(x_train,y_train)
y_pred = final_model_random_forest.predict(x_test)
confusion_mat = confusion_matrix(y_test,y_pred)
roc = roc_auc_score(y_test,y_pred)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

from sklearn import metrics
fpr,tpr, thres = metrics.roc_curve(y_test,y_pred)
import matplotlib.pyplot as plt
plt.title('Receiver operating characterstic curve')
plt.plot(fpr,tpr)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

    
    
# 0.741260470945

#             precision    recall  f1-score   support

#          0       0.79      0.80      0.80     38150
#          1       0.66      0.63      0.65     22494

# avg / total       0.74      0.74      0.74     60644



# Recommendations using Tensorflow, knowledge vault  

#import tensorflow as tf
#
#x = tf.placeholder(dtype=tf.float64,shape=[None,])

# could not be implemented because of memory issues 

#from sklearn.feature_extraction.text import TfidfVectorizer
#tv = TfidfVectorizer(analyzer='word',ngram_range=(1,3),stop_words='english')
#
#d = data1['question1'].astype('U')
##e = data1['question2'].astype('U')
##f = d+e
#tfidf = tv.fit_transform(f)
#featurs = tv.get_feature_names()
#featurs[10:50]
#d2 = (d1*d1.T).toarray()
#
#e1 = tv.transform(e)
#e2 = (e1*e1.T).toarray()
#
#print(e1)











