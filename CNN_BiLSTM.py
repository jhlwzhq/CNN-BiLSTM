#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/razor08/Network-IDS-Paper/blob/master/NSL-KDD-Categorical/NSL_KDD_Multi_Category_k%3D6.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[35]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from datetime import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from joblib import dump, load
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/dataset'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[36]:


import pandas as pd
import numpy as np
import sys
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn import metrics
from scipy.stats import zscore
from tensorflow.keras.utils import get_file, plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
print(pd.__version__)
print(np.__version__)
print(sys.version)
print(sklearn.__version__)


dataset_name='wadi6'

output_file = 'log/'+datetime.now().strftime("%d_%m_%Y_") + 'Results_'+dataset_name+'.csv'

train_file='dataset/'+dataset_name+'train.csv'
test_file='dataset/'+dataset_name+'test.csv'



# In[37]:


#Loading training set into dataframe
train = pd.read_csv(train_file, header=None)
# with open(output_file, "a") as file:
#     file.write('train_file, {} \n '.format(train_file))


# In[38]:


#Loading testing set into dataframe
test = pd.read_csv(test_file, header=None)
# with open(output_file, "a") as file:
#     file.write('test_file, {} \n '.format(test_file))
#
#
# with open(output_file, "a") as file:
#     file.write('training time,testing time,accuracy,precision,recall,f1,FPR,FNR\n')

# In[49]:


#Merging train and test data
combined_data = pd.concat([train,test])
combined_data=combined_data.values
train_n=train.shape[0]
test_n=test.shape[0]
feature_n=train.shape[1]-1



# In[53]:

from sklearn import preprocessing
#Normalizing training set
min_max_scaler = preprocessing.MinMaxScaler()
new_train_df = min_max_scaler.fit_transform(combined_data[:,:combined_data.shape[1]-1])
lb=combined_data[:,combined_data.shape[1]-1]
lb=lb[:, np.newaxis]
new_train_df=np.concatenate([new_train_df,lb],axis=1)


# In[59]:


combined_data_Y=new_train_df[:,new_train_df.shape[1]-1]



# In[61]:


combined_data_X = new_train_df[:,:new_train_df.shape[1]-1]




# In[63]:


from sklearn.model_selection import StratifiedKFold


# In[64]:


kfold = StratifiedKFold(n_splits=6,shuffle=True,random_state=42)
kfold.get_n_splits(combined_data_X,combined_data_Y)


# In[65]:
import time
t1=time.time()


#Bidirectional RNN
batch_size = 32
model = Sequential()
model.add(Convolution1D(64, kernel_size=64, border_mode="same",activation="relu",input_shape=(feature_n, 1)))
model.add(MaxPooling1D(pool_length=(3)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64, return_sequences=False))) 
model.add(Reshape((128, 1), input_shape = (128, )))

model.add(MaxPooling1D(pool_length=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=False))) 

model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[66]:


for layer in model.layers:
    print(layer.output_shape)


# In[67]:


model.summary()


# In[68]:
import csv


# for train_index, test_index in kfold.split(combined_data_X,combined_data_Y):
#     train_x, test_x = combined_data_X[train_index], combined_data_X[test_index]
#     train_y, test_y = combined_data_Y[train_index], combined_data_Y[test_index]
#
#     print("train index:",train_index)
#     print("test index:",test_index)
train_x=combined_data_X[:train_n,:]
test_x=combined_data_X[train_n:,:]
train_y=combined_data_Y[:train_n]
test_y=combined_data_Y[train_n:]

# train_y=train_y[:,np.newaxis]
# test_y=test_y[:,np.newaxis]
# train_out=np.concatenate([train_x,train_y],axis=1)
# test_out=np.concatenate([test_x,test_y],axis=1)
# f1 = open('data/'+dataset_name+'_train.csv','w', newline='')
# writer = csv.writer(f1)
# for i in train_out:
#     writer.writerow(i)
# f1.close()
# f2 = open('data/'+dataset_name+'_test.csv','w', newline='')
# writer = csv.writer(f2)
# for i in test_out:
#     writer.writerow(i)
# f2.close()


train_x=train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
test_x=test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

t1=time.time()
model.fit(train_x, train_y,validation_data=(test_x,test_y), epochs=10)
t2=time.time()
print("Training time:{}".format(t2-t1))
with open(output_file, "a") as file:
    file.write('{},'.format(t2-t1))

t3=time.time()
pred = model.predict(test_x)
pred = np.argmax(pred, axis=1)

score = metrics.accuracy_score(test_y, pred)
print("Validation score: {}".format(score))
t4=time.time()
print("Testing time:{}".format(t4-t3))
with open(output_file, "a") as file:
    file.write('{},'.format(t4-t3))



# In[72]:


from sklearn.metrics import confusion_matrix


# In[73]:


cm = confusion_matrix(test_y, pred, labels=[0,1])


# In[74]:

TN=cm[0,0]
FP=cm[0,1]
FN=cm[1,0]
TP=cm[1,1]

accuracy=(TP+TN)/(TP+TN+FP+FN)
if TP+FP!=0:
    precision=TP/(TP+FP)
else:
    precision=-1.0
if TP+FN!=0:
    recall = TP / (TP + FN)
else:
    recall=-1.0
if precision+recall!=0:
    f1=2*precision*recall/(precision+recall)
else:
    f1=-1.0
if FP + TN!=0:
    FPR = FP / (FP + TN)
else:
    FPR=-1.0
if TP+TN!=0:
    FNR=FN/(TP+TN)
else:
    FNR=-1.0

print('accuracy:{:.4},precision:{:.4},recall:{:.4},f1:{:.4},FPR:{:.4},FNR:{:.4}'.format(accuracy,precision,recall,f1,FPR,FNR))
with open(output_file, "a") as file:
    file.write('{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n'.format(accuracy*100,precision*100,recall*100,f1,FPR*100,FNR*100))


print(cm)
# with open(output_file, "a") as file:
#     file.write('cm,{}\n\n\n\n'.format(cm))


