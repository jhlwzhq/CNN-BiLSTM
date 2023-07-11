# 用于工业物联网异常检测的CNN-BiLSTM模型

## 数据集
本项目使用工业异常检测数据集[SWaT](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat/)和[WADI](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_wadi/)进行验证，数据集可点击链接至官网下载  
DataSet文件夹中给出了用于分割训练集和测试集的代码split_dataset.py

## 模型
CNN_BiLSTM.py中给出了模型的训练及测试代码，训练结果以CSV文件的形式保存至当前目录的log文件夹下