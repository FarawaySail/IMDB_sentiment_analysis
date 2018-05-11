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

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return(meaningful_words) 

word_list = []
for i in range(25000):
    if((i+1) % 100 == 0):
        print("Review %d of %d\n" % ( i+1, 25000 ))
    word_list = word_list + review_to_words(train["review"][i])


word_set = set(word_list)


with open("word_set.txt","w") as file1:
    file1.write(str(word_set))
with open("word_list.txt","w") as file2:
    file2.write(str(word_list))

train_x = []
for i in range(25000):
    if((i+1) % 100 == 0):
        print("Review %d of %d\n" % ( i+1, 25000 ))
    temp = []
    for x in range(len(review_to_words(train["review"][i]))):
        temp.append(list(word_set).index(review_to_words(train["review"][i])[x]))
    train_x.append(temp)

y_train = train['sentiment'][0:20000]
y_test  = train['sentiment'][20000:25000]
print(train_x)
train_x=sequence.pad_sequences(train_x, maxlen=100)
with open("train_X.csv","w",newline='') as file3:
    csv_write = csv.writer(file3)
    for i in train_x:
        csv_write.writerow(i)

with open("train_X.csv","r") as file4:
    csv_reader=csv.reader(file4,dialect='excel')
    x=[]
    for i in file4:
        i=i.replace('\n','')
        xi=i.split(',')
        xi=[int(xi) for xi in xi if xi]
        x.append(xi)
print(x)
x_test = x[20000:25000]
x_train = x[0:20000]
print(x_test,x_train)
len(x_test)

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


