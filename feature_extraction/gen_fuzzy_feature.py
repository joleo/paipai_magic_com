# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_fuzzy_feature
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

# question = pd.read_csv('../data/question.csv')
# question = question[['qid','chars']]
#
# train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
# train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
# train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
# train = train[['label','chars_x','chars_y']]
# train.columns = ['label','question1','question2']
# # len_train = train.shape[0]
# len_train = len(train)
#
# test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
# test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
# test = test[['chars_x','chars_y']]
# test.columns = ['question1','question2']
#
# df_feat = pd.DataFrame()
# df_data = pd.concat([train,test])
#
# # 输出相似度的结果
# # https://blog.csdn.net/sunyao_123/article/details/76942809
# df_feat['fuzz_qratio'] = df_data.apply(lambda row: fuzz.QRatio(str(row['question1']), str(row['question2'])), axis=1)
# df_feat['fuzz_WRatio'] = df_data.apply(lambda row: fuzz.WRatio(str(row['question1']), str(row['question2'])), axis=1)
# df_feat['fuzz_partial_ratio'] = df_data.apply(lambda row: fuzz.partial_ratio(str(row['question1']), str(row['question2'])), axis=1)
# df_feat['fuzz_partial_token_set_ratio'] = df_data.apply(lambda row: fuzz.partial_token_set_ratio(str(row['question1']), str(row['question2'])), axis=1)
# df_feat['fuzz_partial_token_sort_ratio'] = df_data.apply(lambda row: fuzz.partial_token_sort_ratio(str(row['question1']), str(row['question2'])), axis=1)
# df_feat['fuzz_token_set_ratio'] = df_data.apply(lambda row: fuzz.token_set_ratio(str(row['question1']), str(row['question2'])), axis=1)
# df_feat['fuzz_token_sort_ratio'] = df_data.apply(lambda row: fuzz.token_sort_ratio(str(row['question1']), str(row['question2'])), axis=1)
#
# # python-Levenshtein
# path = '../data/pre_feature/'
# df_feat[:len_train].to_csv(path + 'train_feature_fuzz.csv', index=False)
# df_feat[len_train:].to_csv(path + 'test_feature_fuzz.csv', index=False)

question = pd.read_csv('../data/question.csv')
question = question[['qid','words']]

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
df_data = pd.concat([train,test])

# 输出相似度的结果
# https://blog.csdn.net/sunyao_123/article/details/76942809
df_feat['fuzz_words_qratio'] = df_data.apply(lambda row: fuzz.QRatio(str(row['words1']), str(row['words2'])), axis=1)
df_feat['fuzz_words_WRatio'] = df_data.apply(lambda row: fuzz.WRatio(str(row['words1']), str(row['words2'])), axis=1)
df_feat['fuzz_words_partial_ratio'] = df_data.apply(lambda row: fuzz.partial_ratio(str(row['words1']), str(row['words2'])), axis=1)
df_feat['fuzz_words_partial_token_set_ratio'] = df_data.apply(lambda row: fuzz.partial_token_set_ratio(str(row['words1']), str(row['words2'])), axis=1)
df_feat['fuzz_words_partial_token_sort_ratio'] = df_data.apply(lambda row: fuzz.partial_token_sort_ratio(str(row['words1']), str(row['words2'])), axis=1)
df_feat['fuzz_words_token_set_ratio'] = df_data.apply(lambda row: fuzz.token_set_ratio(str(row['words1']), str(row['words2'])), axis=1)
df_feat['fuzz_words_token_sort_ratio'] = df_data.apply(lambda row: fuzz.token_sort_ratio(str(row['words1']), str(row['words2'])), axis=1)

# python-Levenshtein
path = '../data/pre_feature/'
df_feat[:len_train].to_csv(path + 'train_feature_fuzz.csv', index=False)
df_feat[len_train:].to_csv(path + 'test_feature_fuzz.csv', index=False)