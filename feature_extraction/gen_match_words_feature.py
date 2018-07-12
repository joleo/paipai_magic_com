# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_match_feature
   Description :
   Author :       Administrator
   date：          2018/6/18 0018
-------------------------------------------------
   Change Activity:
                   2018/6/18 0018:
-------------------------------------------------
"""
__author__ = 'Administrator'

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')

# stop_word = ['L1187','L1368','L0362','L1128','LL32','L0143','L3019','L2218','L2582','L1861'
#              ,'L2214','L0104']

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

df_feat = pd.DataFrame()
df_data = pd.concat([train,test], axis=0)

stops = set(stopwords.words('english'))
# stops = stop_word
def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


df_feat['word_match_word_ratio'] = df_data.apply(word_match_share, axis=1)

# Unigrams without stops
def get_unigrams(que):
    return [word for word in nltk.word_tokenize(que.lower()) if word not in stops]
df_data["unigrams_nostop_q1"] = df_data['question1'].apply(lambda x: get_unigrams(str(x)))
df_data["unigrams_nostop_q2"] = df_data['question2'].apply(lambda x: get_unigrams(str(x)))

def get_word_match_nostop_unigrams_count(row):
    return len(set(row["unigrams_nostop_q1"]).intersection(set(row["unigrams_nostop_q2"])))

def get_word_match_nostop_unigrams_ratio(row):
    return float(row["unigrams_nostop_match_count"])/max(len(set(row["unigrams_nostop_q1"]).union(set(row["unigrams_nostop_q2"]))),1)

unigrams_nostop_match_count = []
unigrams_nostop_match_ratio = []

for index, row in df_data.iterrows():
    count = len(set(row["unigrams_nostop_q1"]).intersection(set(row["unigrams_nostop_q2"])))
    ratio = float(count) / max(len(set(row["unigrams_nostop_q1"]).union(set(row["unigrams_nostop_q2"]))),1)
    unigrams_nostop_match_count.append(count)
    unigrams_nostop_match_ratio.append(ratio)

df_feat["unigrams_nostop_match_word_count"] = unigrams_nostop_match_count
df_feat["unigrams_nostop_match_word_ratio"] = unigrams_nostop_match_ratio


# Bigrams without stops
# ngrams方法
def get_bigrams(que):
    return [i for i in nltk.ngrams(que, 2)]
df_data["bigrams_nostop_q1"] = df_data["unigrams_nostop_q1"].apply(lambda x: get_bigrams(x))
df_data["bigrams_nostop_q2"] = df_data["unigrams_nostop_q2"].apply(lambda x: get_bigrams(x))

bigrams_nostop_match_count = []
bigrams_nostop_match_ratio = []

for index, row in df_data.iterrows():
    count = len(set(row["bigrams_nostop_q1"]).intersection(set(row["bigrams_nostop_q2"])))
    ratio = float(count) / max(len(set(row["bigrams_nostop_q1"]).union(set(row["bigrams_nostop_q2"]))),1)
    bigrams_nostop_match_count.append(count)
    bigrams_nostop_match_ratio.append(ratio)

df_feat["bigrams_nostop_match_word_count"] = bigrams_nostop_match_count
df_feat["bigrams_nostop_match_word_ratio"] = bigrams_nostop_match_ratio


# Trigrams without stops
def get_trigrams(que):
    return [i for i in nltk.ngrams(que, 3)]
df_data["trigrams_nostop_q1"] = df_data["unigrams_nostop_q1"].apply(lambda x: get_trigrams(x))
df_data["trigrams_nostop_q2"] = df_data["unigrams_nostop_q2"].apply(lambda x: get_trigrams(x))

trigrams_nostop_match_count = []
trigrams_nostop_match_ratio = []

for index, row in df_data.iterrows():
    count = len(set(row["trigrams_nostop_q1"]).intersection(set(row["trigrams_nostop_q2"])))
    ratio = float(count) / max(len(set(row["trigrams_nostop_q1"]).union(set(row["trigrams_nostop_q2"]))),1)
    trigrams_nostop_match_count.append(count)
    trigrams_nostop_match_ratio.append(ratio)

df_feat["trigrams_nostop_match_word_count"] = trigrams_nostop_match_count
df_feat["trigrams_nostop_match_word_ratio"] = trigrams_nostop_match_ratio


path = '../data/pre_feature/'
df_feat[:len_train].to_csv(path + 'train_feature_word_match.csv', index=False)
df_feat[len_train:].to_csv(path + 'test_feature_word_match.csv', index=False)

