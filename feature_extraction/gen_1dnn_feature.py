# -*- coding: utf-8 -*-
# @Time    : 6/21/18 6:32 PM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_1dnn_feature.py
import numpy as np
import pandas as pd
import gensim
from gensim.models.wrappers import FastText
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import _save, _load, SaveData

BASE_DIR = '../data/'
EMBEDDING_FILE = '../data/char_embed.txt'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 25
MAX_NB_WORDS = 1000 # 处理的最大单词数量
EMBEDDING_DIM = 300


print('Processing text dataset')
# word level
question = pd.read_csv('../data/question.csv')
question = question[['qid','words']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','words_x','words_y']]
train.columns = ['label','question1','question2']
test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['words_x','words_y']]
test.columns = ['question1','question2']
path = '../data/pre_feature/'


texts_1 = []
texts_2 = []
labels = []
def get_text(row):
    global texts_1, texts_2, labels
    texts_1.append(row.question1)
    texts_2.append(row.question2)
    labels.append(int(row.label))
train.apply(get_text, axis=1)
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
# test_ids = []
def get_test_text(row):
    global test_texts_1, test_texts_2
    test_texts_1.append(row.question1)
    test_texts_2.append(row.question2)
    # test_ids.append(row.test_id)
test.apply(get_test_text, axis=1)
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
# test_ids = np.array(test_ids)

print('Preparing embedding matrix')
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


# char
embeddings_index = {}
with open('../data/char_embed.txt','r') as f:
    for i in f:
        values = i.split(' ')
        word = str(values[0])
        embedding = np.asarray(values[1:],dtype='float')
        embeddings_index[word] = embedding
print('char embedding',len(embeddings_index)) # 20891

#
EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS,len(word_index))
char_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(str(word).upper())
    if embedding_vector is not None:
        char_embedding_matrix[i] = embedding_vector

path = '../data/'
save_data = SaveData()
save_data.data_1 = data_1
save_data.data_2 = data_2
# save_data.magic_train_dense = magic_train_dense
save_data.labels = labels
save_data.test_data_1 = test_data_1
save_data.test_data_2 = test_data_2
# save_data.magic_test_dense = magic_test_dense
# save_data.test_ids = test_ids
save_data.embedding_matrix = word_embedding_matrix
save_data.char_embedding_matrix = char_embedding_matrix
save_data.nb_words = nb_words
import cPickle
cPickle.dump(save_data,open(path + 'nn_glove_embedding_wc_data.pkl',"wb"))