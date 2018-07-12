# -*- coding: utf-8 -*-
# @Time    : 6/20/18 8:15 PM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_simhash_feature.py

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from simhash import Simhash

"""
SimHash 特征（基于 simhash）:

word/word bigrams/word trigrams distance
character bigrams/character trigrams distance
"""
question = pd.read_csv('../data/question.csv')
question = question[['qid','words','chars']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','words_x','words_y']]
train.columns = ['label','words1','words2']
len_train = train.shape[0]

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['words_x','words_y']]
test.columns = ['words1','words2']

df_feat = pd.DataFrame()
df_data = pd.concat([train[['words1', 'words2']], test[['words1', 'words2']]], axis=0)


def tokenize(sequence):
    words = word_tokenize(sequence)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words

def clean_sequence(sequence):
    tokens = tokenize(sequence)
    return ' '.join(tokens)

def get_word_ngrams(sequence, n=3):
    tokens = tokenize(sequence)
    return [' '.join(ngram) for ngram in ngrams(tokens, n)]

def get_character_ngrams(sequence, n=3):
    sequence = clean_sequence(sequence)
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]


def caluclate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))

def get_word_distance(questions):
    # 词距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return caluclate_simhash_distance(q1, q2)

def get_word_2gram_distance(questions):
    # word_ngrams距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_char_2gram_distance(questions):
    # char_2gram距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_word_3gram_distance(questions):
    # word_3gram距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)

def get_char_3gram_distance(questions):
    # char_3gram距离
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)


df_data['words'] = df_data['words1'] + '_split_tag_' + df_data['words2']


df_feat['simhash_tokenize_distance'] = df_data['words'].apply(get_word_distance)
df_feat['simhash_word_2gram_distance'] = df_data['words'].apply(get_word_2gram_distance)
df_feat['simhash_char_2gram_distance'] = df_data['words'].apply(get_char_2gram_distance)
df_feat['simhash_word_3gram_distance'] = df_data['words'].apply(get_word_3gram_distance)
df_feat['simhash_char_3gram_distance'] =df_data['words'].apply(get_char_3gram_distance)

path = '../data/pre_feature/'
df_feat[:len_train].to_csv(path + 'train_feature_simhash.csv', index=False)
df_feat[len_train:].to_csv(path + 'test_feature_simhash.csv', index=False)