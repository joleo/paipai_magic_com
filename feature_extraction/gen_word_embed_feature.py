# -*- coding: utf-8 -*-
# @Time    : 6/21/18 2:04 AM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_word_embed_feature.py
import pandas as pd
import numpy as np
import gensim
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from gensim.models.wrappers import FastText


question = pd.read_csv('../data/question.csv')
question = question[['qid','words','chars']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
# train = train[['label','words_x','words_y']]
# train.columns = ['label','words1','words2']
train = train[['label','chars_x','chars_y']]
train.columns = ['label','chars1','chars2']
len_train = train.shape[0]

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
# test = test[['words_x','words_y']]
# test.columns = ['words1','words2']
test = test[['chars_x','chars_y']]
test.columns = ['chars1','chars2']

df_feat = pd.DataFrame()
# df_data = pd.concat([train[['words1', 'words2']], test[['words1', 'words2']]], axis=0)
df_data = pd.concat([train[['chars1', 'chars2']], test[['chars1', 'chars2']]], axis=0)


model = gensim.models.KeyedVectors.load_word2vec_format('../data/char_embed.txt', binary=False, encoding='utf8')

# def wmd(s1, s2):
#     s1 = str(s1).lower().split()
#     s2 = str(s2).lower().split()
#     stop_words = stopwords.words('english')
#     s1 = [w for w in s1 if w not in stop_words]
#     s2 = [w for w in s2 if w not in stop_words]
#     return model.wmdistance(s1, s2)
#
# df_feat['glove_wmd'] = df_data.apply(lambda row: wmd(row['chars1'], row['chars2']), axis=1)









#
# path = '../data/pre_feature/'
# df_feat[:len_train].to_csv(path + 'train_feature_embedding.csv', index=False)
# df_feat[len_train:].to_csv(path + 'test_feature_embedding.csv', index=False)
