# -*- coding: utf-8 -*-
# @Time    : 6/20/18 8:33 PM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_tfidf_feature.py
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


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


# stop_word = ['L1187','L1368','L0362','L1128','LL32','L0143','L3019','L2218','L2582','L1861'
#              ,'L2214','L0104']
# stops = stop_word #set(stopwords.words(stop_word))
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

questions_txt = pd.Series(
    df_data['words1'].tolist() +
    df_data['words2'].tolist()
).astype(str)

tfidf.fit_transform(questions_txt)

tfidf_sum1 = []
tfidf_sum2 = []
tfidf_mean1 = []
tfidf_mean2 = []
tfidf_len1 = []
tfidf_len2 = []

for index, row in df_data.iterrows():
    tfidf_q1 = tfidf.transform([str(row['words1'])]).data
    tfidf_q2 = tfidf.transform([str(row['words2'])]).data

    tfidf_sum1.append(np.sum(tfidf_q1))
    tfidf_sum2.append(np.sum(tfidf_q2))
    tfidf_mean1.append(np.mean(tfidf_q1))
    tfidf_mean2.append(np.mean(tfidf_q2))
    tfidf_len1.append(len(tfidf_q1))
    tfidf_len2.append(len(tfidf_q2))

df_feat['tfidf_sum1'] = tfidf_sum1
df_feat['tfidf_sum2'] = tfidf_sum2
df_feat['tfidf_mean1'] = tfidf_mean1
df_feat['tfidf_mean2'] = tfidf_mean2
df_feat['tfidf_len1'] = tfidf_len1
df_feat['tfidf_len2'] = tfidf_len2

path = '../data/pre_feature/'
df_feat.fillna(0.0)
df_feat[:len_train].to_csv(path + 'train_feature_tfidf.csv', index=False)
df_feat[len_train:].to_csv(path + 'test_feature_tfidf.csv', index=False)




