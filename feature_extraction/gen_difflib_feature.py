# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_difflib_feature
   Description :
   Author :       Administrator
   date：          2018/6/17 0017
-------------------------------------------------
   Change Activity:
                   2018/6/17 0017:
-------------------------------------------------
"""
__author__ = 'Administrator'

import difflib
import pandas as pd
import numpy as np

question = pd.read_csv('../data/question.csv')
question = question[['qid','chars']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','chars_x','chars_y']]
train.columns = ['label','chars1','chars2']
len_train = len(train)

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['chars_x','chars_y']]
test.columns = ['chars1','chars2']


all = pd.concat([train,test], sort=False)

def diff_ratios(row):
    # 有效问题的个数比例
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(row.chars1).lower(), str(row.chars2).lower())
    return seq.ratio() # 返回一个度量两个序列的相似程度，值在[0, 1]之间

df_feat = pd.DataFrame()

# 两个问题的相似程度特征
df_feat['diff_ratios'] = all[['chars1','chars2']].apply(diff_ratios, axis=1)

path = '../data/pre_feature/'
df_feat[:len_train].to_csv(path + 'train_feature_difflib.csv', index=False)
df_feat[len_train:].to_csv(path + 'test_feature_difflib.csv', index=False)





# question = pd.read_csv('../data/question.csv')
# question = question[['qid','words']]
#
# train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
# train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
# train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
# train = train[['label','words_x','words_y']]
# train.columns = ['label','words1','words2']
# len_train = train.shape[0]
#
# test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
# test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
# test = test[['words_x','words_y']]
# test.columns = ['words1','words2']
#
#
# all = pd.concat([train,test], sort=False)
#
# def diff_ratios(row):
#     # 有效问题的个数比例
#     seq = difflib.SequenceMatcher()
#     seq.set_seqs(str(row.words1).lower(), str(row.words2).lower())
#     return seq.ratio() # 返回一个度量两个序列的相似程度，值在[0, 1]之间
#
# df_feat = pd.DataFrame()
#
# # 两个问题的相似程度特征
# df_feat['diff_words_ratios'] = all[['words1','words2']].apply(diff_ratios, axis=1)
#
# path = '../data/pre_feature/'
# df_feat[:len_train].to_csv(path + 'train_feature_words_difflib.csv', index=False)
# df_feat[len_train:].to_csv(path + 'test_feature_words_difflib.csv', index=False)