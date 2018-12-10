import pandas as pd
import numpy as np
from keras.models import Model,Sequential,load_model
from keras.layers import Input,Embedding,Dense,LSTM,Dropout
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('train.csv', sep='\t')

X = dataset['Phrase']
Y = dataset['Sentiment']

Y = Y.reshape([-1,1])
one_hot = OneHotEncoder(sparse=False)
Y_oneh = one_hot.fit_transform(Y)

words_list=[]
for  i in range(X.shape[0]):
  t_list = X[i].split(" ")
  for each_word in t_list:
    words_list.append(each_word)

vocab_list = list(set(words_list))

words_indices={}
indices_words = {}
for i,word in enumerate(vocab_list):
  words_indices[word] = i
  indices_words[i] = word
  

X_num = np.zeros((X.shape[0],100))
for  i in range(X.shape[0]):
  words = X[i].split(" ")
  for j,each_w in enumerate(words):
    X_num[i,j]= words_indices[each_w]
    
in_M = Input(shape=(100,))
M =  Embedding(input_dim=len(vocab_list),output_dim=300)(in_M)
M = LSTM(64,return_sequences=True)(M)
M = Dropout(0.5)(M)
M = LSTM(64,return_sequences=False)(M)
M = Dropout(0.5)(M)
M_out = Dense(5,activation='softmax')(M)


model = Model(inputs=in_M, outputs=M_out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_num,Y_oneh,epochs=10,batch_size=32)

model.save('senti.h5')

x = X_num[5,:]
x = x.reshape([100,])
y_out = model1.predict(X_num[226:227,:])

sentiment = np.argmax(y_out)
