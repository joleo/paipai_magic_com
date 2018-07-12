# -*- coding: utf-8 -*-
# @Time    : 6/20/18 10:35 PM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_graph_freq_feature.py
import numpy as np
import pandas as pd
import timeit

question = pd.read_csv('../data/question.csv')
question = question[['qid','words','chars']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','words_x','words_y']]
train.columns = ['label','question1','question2']
len_train = train.shape[0]

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['words_x','words_y']]
test.columns = ['question1','question2']



tic0=timeit.default_timer()
df1 = train[['question1']].copy()
df2 = train[['question2']].copy()
df1_test = test[['question1']].copy()
df2_test = test[['question2']].copy()

df2.rename(columns = {'question2':'question1'},inplace=True)
df2_test.rename(columns = {'question2':'question1'},inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
#train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
train_questions.drop_duplicates(subset = ['question1'],inplace=True)

train_questions.reset_index(inplace=True,drop=True)
questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
train_cp = train.copy()
test_cp = test.copy()
# train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

test_cp['label'] = -1
# test_cp.rename(columns={'test_id':'id'},inplace=True)
comb = pd.concat([train_cp,test_cp])

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()

def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
#map to frequency space
comb['q1_word_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
comb['q2_word_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

train_comb = comb[comb['label'] >= 0][['q1_hash','q2_hash','q1_word_freq','q2_word_freq','label']]
test_comb = comb[comb['label'] < 0][['q1_hash','q2_hash','q1_word_freq','q2_word_freq']]



path = '../data/pre_feature/'
train_comb[['q1_word_freq', 'q2_word_freq']].to_csv(path + 'train_feature_graph_freq.csv', index=False)
test_comb[['q1_word_freq', 'q2_word_freq']].to_csv(path + 'test_feature_graph_freq.csv', index=False)




