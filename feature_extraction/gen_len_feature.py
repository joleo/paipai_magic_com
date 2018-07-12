# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_len_feature
   Description :
   Author :       Administrator
   date：          2018/6/17 0017
-------------------------------------------------
   Change Activity:
                   2018/6/17 0017:
-------------------------------------------------
"""
__author__ = 'Administrator'
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np

question = pd.read_csv('../data/question.csv')
question = question[['qid','words','chars']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','words_x','words_y','chars_x','chars_y']]
train.columns = ['label','words1','words2','chars1','chars2']
len_train = train.shape[0]

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['words_x','words_y','chars_x','chars_y']]
test.columns = ['words1','words2','chars1','chars2']

df_feat = pd.DataFrame()
df_data = pd.concat([train,test])

# generate len features
# 长度特征
df_feat['word_len1'] = df_data.words1.map(lambda x: len(str(x))) # word长度
df_feat['word_len2'] = df_data.words2.map(lambda x: len(str(x)))

df_feat['char_len1'] = df_data.chars1.map(lambda x: len(str(x))) # 字符长度
df_feat['char_len2'] = df_data.chars2.map(lambda x: len(str(x)))

# 差值特征
df_feat['word_len_diff_ratio'] = df_feat.apply(
    lambda row: abs(row.word_len1-row.word_len2)/(row.word_len1+row.word_len2), axis=1)
df_feat['char_len_diff_ratio'] = df_feat.apply(
    lambda row: abs(row.char_len1-row.char_len2)/(row.char_len1+row.char_len2), axis=1)

path = '../data/pre_feature/'
df_feat[:len_train].to_csv(path + 'train_feature_len.csv', index=False)
df_feat[len_train:].to_csv(path + 'test_feature_len.csv', index=False)


