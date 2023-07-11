import pandas as pd  # 数据科学计算工具
import numpy as np  # 数值计算工具
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from sklearn.model_selection import train_test_split  # 将数据集分开成训练集和测试集
from sklearn import preprocessing
import csv


file_path='skindata.csv'
print('file:'+file_path)

dataset = pd.read_csv(file_path, header=None)
dataset=dataset.values
m, n = dataset.shape
print('total_num:{}'.format(m))
label=dataset[:,n-1]
features=dataset[:,:n-1]

self = features[dataset[:, n-1] == 0, :]
m_self,n_self=self.shape
print('self_num:{}'.format(m_self))
print('nonself_num:{}'.format(m-m_self))
train_num=m_self*0.8

# standard_scaler = preprocessing.StandardScaler(with_mean=False)
# features = pd.DataFrame(standard_scaler.fit_transform(features))
train_X, test_X, train_lb, test_lb = train_test_split(features,
                                                      label,
                                                      train_size=int(train_num),
                                                      random_state=1)
train_lb=train_lb[:,np.newaxis]
train=np.concatenate((train_X, train_lb),axis=1)
test_lb=test_lb[:,np.newaxis]
test=np.concatenate((test_X, test_lb),axis=1)
print('train_num:{}'.format(train.shape[0]))
print('test_num:{}'.format(test.shape[0]))


f1 = open('skindatatrain.csv','w', newline='')
writer = csv.writer(f1)
for i in train:
    writer.writerow(i)
f1.close()

f2 = open('skindatatest.csv','w', newline='')
writer = csv.writer(f2)
for i in test:
    writer.writerow(i)
f2.close()