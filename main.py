import pandas as pd
from bs4 import BeautifulSoup
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import re
import csv
from nltk.corpus import stopwords

train = pd.read_csv("./labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

y_train = train['sentiment'][0:20000]
y_test  = train['sentiment'][20000:25000]

with open("./train_X.csv","r") as file4:
    csv_reader=csv.reader(file4,dialect='excel')
    x=[]
    for i in file4:
        i=i.replace('\n','')
        xi=i.split(',')
        xi=[int(xi) for xi in xi if xi]
        x.append(xi)

x_test = x[20000:25000]
x_train = x[0:20000]

print('Build model...')
model = Sequential()
model.add(Embedding(80000, 128, mask_zero=True))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(np.array(x_test), np.array(y_test),
          batch_size=32,
          epochs=10,
          validation_data=(np.array(x_test), np.array(y_test)))

score, acc = model.evaluate(np.array(x_test), np.array(y_test),
                            batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)


