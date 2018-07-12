# -*- coding: utf-8 -*-
# @Time    : 6/20/18 11:07 PM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_graph_interect_feature.py
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.corpus import stopwords


question = pd.read_csv('../data/question.csv')
question = question[['qid','words']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','words_x','words_y']]
train.columns = ['label','question1','question2']
# len_train = train.shape[0]
len_train = len(train)

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['words_x','words_y']]
test.columns = ['question1','question2']


ques = pd.concat([train[['question1', 'question2']], test[['question1', 'question2']]], axis=0).reset_index(drop='index')


stops = set(stopwords.words("english"))
def word_match_share(q1, q2, stops=None):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stops:
            q1words[word] = 1
    for word in q2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


q_dict = defaultdict(dict)
for i in range(ques.shape[0]):
        wm = word_match_share(ques.question1[i], ques.question2[i], stops=stops)
        q_dict[ques.question1[i]][ques.question2[i]] = wm
        q_dict[ques.question2[i]][ques.question1[i]] = wm


def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))
def q1_q2_wm_ratio(row):
    q1 = q_dict[row['question1']]
    q2 = q_dict[row['question2']]
    inter_keys = set(q1.keys()).intersection(set(q2.keys()))
    if(len(inter_keys) == 0): return 0.
    inter_wm = 0.
    total_wm = 0.
    for q,wm in q1.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    for q,wm in q2.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    if(total_wm == 0.): return 0.
    return inter_wm/total_wm


# train['q1_q2_wm_ratio'] = train.apply(q1_q2_wm_ratio, axis=1, raw=True)
# test['q1_q2_wm_ratio'] = test.apply(q1_q2_wm_ratio, axis=1, raw=True)

train['q1_q2_intersect'] = train.apply(q1_q2_intersect, axis=1, raw=True)
test['q1_q2_intersect'] = test.apply(q1_q2_intersect, axis=1, raw=True)

# train_feat = train[['q1_q2_intersect', 'q1_q2_wm_ratio']]
# test_feat = test[['q1_q2_intersect', 'q1_q2_wm_ratio']]

train_feat = train[[ 'q1_q2_intersect']]
test_feat = test[['q1_q2_intersect']]

path = '../data/pre_feature/'
train_feat.to_csv(path + 'train_feature_graph_interect.csv', index=False)
train_feat.to_csv(path + 'test_feature_graph_interect.csv', index=False)