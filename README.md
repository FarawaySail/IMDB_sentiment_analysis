# 基于LSTM的IMDB二元情感分类
## 任务简介
IMDB二元情感分析是自然语言处理的基本任务，本工程基于labeledTrainData.tsv提供的原始数据，进行数据处理、网络搭建、最后的训练和预测，成功实现了一个正确率接近100%的模型。
## 需要的包和库
> pandas、keras、bs4、numpy、nltk、csv

## 文件清单
data.py——数据处理部分，通过将评价分词、序列化得到待训练的数据  
main.py——网络搭建和训练部分  
word_list.txt——分词后的词表  
word_set.txt——词典  
train_X.csv——训练集所需序列  
IMDB.py——整合版本，从数据处理到训练  
## 网络结构
`
model = Sequential()
model.add(Embedding(80000, 128, mask_zero=True))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))
`
 

