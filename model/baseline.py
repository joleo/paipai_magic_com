# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     baseline
   Description :
   Author :       Administrator
   date：          2018/6/10 0010
-------------------------------------------------
   Change Activity:
                   2018/6/10 0010:
-------------------------------------------------
"""
__author__ = 'Administrator'
#coding:utf-8

'''
简单的baselin
公众号：麻婆豆腐AI
github：zs167275

网络基本结构：

   INPUT1      IINPUT2
    |          |
embedding_q1 embedding_q2
    |          |
    |          |
    ------------
          |
          |
        全连接
          |
         output

感受：本人没有GPU，训练的好慢啊，慢啊，慢啊。
收购二手1080，10块钱无限收

线下随机分割的数据集：
线下logloss 0.3623
线上logloss 0.376738
'''

import pandas as pd
import numpy as np
# 文本处理
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# word level
question = pd.read_csv('../data/question.csv')
question = question[['qid','words']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','words_x','words_y']]
train.columns = ['label','q1','q2']

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['words_x','words_y']]
test.columns = ['q1','q2']

all = pd.concat([train,test])



MAX_NB_WORDS = 1000 # 处理的最大单词数量

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(question['words'])

q1_word_seq = tokenizer.texts_to_sequences(all['q1']) # 要用以训练的文本列表
q2_word_seq = tokenizer.texts_to_sequences(all['q2'])
word_index = tokenizer.word_index
print(len(word_index))# 20890

# 准备好word embedding
embeddings_index = {}
with open('../data/word_embed.txt','r') as f:
    for i in f:
        values = i.split(' ')
        word = str(values[0])
        embedding = np.asarray(values[1:],dtype='float')
        embeddings_index[word] = embedding
print('word embedding',len(embeddings_index)) # 20891

#
EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS,len(word_index))
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(str(word).upper())
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

# https://blog.csdn.net/jiaach/article/details/79403352
'''
RNN网络容易出现反向传播过程中的梯度问题。主要原因是我们通常给RNN的参数为有限的序列
1、为了实现的简便，keras只能接受长度相同的序列输入。因此如果目前序列长度参差不齐，这时需要使用pad_sequences()。
该函数是将序列转化为经过填充以后的一个新序列。
2、举一个例子，是否使用对齐函数取决于如何切割本文，对于一个文本而言，如果是选择根据‘。’来分割句子，
因此需要使用该函数保证每个分割的句子能够得到同等长度，但是更加聪明的做法是考虑将文本按照每一个字来分隔，
3、保证切割的句子都是等长的句子，不要再使用该函数。
4、最后，输入RNN网络之前将词汇转化为分布式表示。
'''
MAX_SEQUENCE_LENGTH = 25 #30
q1_data = pad_sequences(q1_word_seq,maxlen=MAX_SEQUENCE_LENGTH)
q2_data = pad_sequences(q2_word_seq,maxlen=MAX_SEQUENCE_LENGTH)

train_q1_data = q1_data[:train.shape[0]]
train_q2_data = q2_data[:train.shape[0]]

test_q1_data = q1_data[train.shape[0]:]
test_q2_data = q2_data[train.shape[0]:]

labels = train['label']
print('Shape of question1 train data tensor:', train_q1_data.shape)
print('Shape of question2 train data tensor:', train_q2_data.shape)
print('Shape of question1 test data tensor:', test_q1_data.shape)
print('Shape of question1 test data tensor:', test_q2_data.shape)
print('Shape of label tensor:', labels.shape)


#准备好训练集
X = np.stack((train_q1_data, train_q2_data), axis=1)
y = labels

from sklearn.model_selection import StratifiedShuffleSplit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]


# Define the model
DROPOUT = 0.25
question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

q1 = Embedding(nb_words + 1,
                 EMBEDDING_DIM,
                 weights=[word_embedding_matrix],
                 input_length=MAX_SEQUENCE_LENGTH,
                 trainable=False)(question1)
# 将词汇转化为分布式表示
q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)

q2 = Embedding(nb_words + 1,
                 EMBEDDING_DIM,
                 weights=[word_embedding_matrix],
                 input_length=MAX_SEQUENCE_LENGTH,
                 trainable=False)(question2)
q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)

# input layer data
merged = concatenate([q1,q2])

# layer1
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

# full connection
is_duplicate = Dense(1, activation='sigmoid')(merged)

# fitting model
model = Model(inputs=[question1,question2], outputs=is_duplicate)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'])

history = model.fit([Q1_train, Q2_train],
                    y_train,
                    epochs=6,
                    validation_data=[[Q1_test,Q2_test],y_test],
                    verbose=1,
                    batch_size=128,
                    )
# model predict
result = model.predict([test_q1_data,test_q2_data],batch_size=1024)

# submit result
submit = pd.DataFrame()
submit['y_pre'] = list(result[:,0])
submit.to_csv('../data/baseline.csv',index=False)
#
# # submit result
# submit_history = pd.DataFrame()
# submit_history['train'] = list(history[:,0])
# submit_history.to_csv('../data/train.csv',index=False)