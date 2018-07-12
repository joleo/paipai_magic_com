# -*- coding: utf-8 -*-
# @Time    : 6/25/18 4:45 AM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : model_xgb.py
import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.cross_validation import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
from utils import _load, _save
from sklearn.metrics import log_loss
import xgboost as xgb
seed=1024
np.random.seed(seed)

path = '../data/pre_feature/'
train_difflib = pd.read_csv(path + 'train_feature_difflib.csv')
train_word_difflib = pd.read_csv(path + 'train_feature_words_difflib.csv')
# train_embedding = pd.read_csv('train_feature_embedding.csv')
train_fuzz = pd.read_csv(path + 'train_feature_fuzz.csv')
train_len = pd.read_csv(path + 'train_feature_len.csv')
train_match = pd.read_csv(path + 'train_feature_match.csv')
train_word_match = pd.read_csv(path + 'train_feature_word_match.csv')

train_match_2 = pd.read_csv(path + 'train_feature_match_2.csv')
# train_oof = pd.read_csv('train_feature_oof.csv')
train_simhash = pd.read_csv(path + 'train_feature_simhash.csv')
train_tfidf = pd.read_csv(path + 'train_feature_tfidf.csv')
train_ngram = pd.read_csv(path + 'train_ngram_feature.csv')
train_word_ngram = pd.read_csv(path + 'train_word_ngram_feature.csv')

# train_graph_clique = pd.read_csv('train_feature_graph_clique.csv')
train_graph_chars_pagerank = pd.read_csv(path + 'train_feature_graph_chars_pagerank.csv')
train_graph_pagerank = pd.read_csv(path + 'train_feature_graph_pagerank.csv')
train_graph_chars_freq = pd.read_csv(path +'train_feature_graph_chars_freq.csv')
train_graph_freq = pd.read_csv(path +'train_feature_graph_freq.csv')
# train_graph_intersect = pd.read_csv(path + 'train_feature_graph_intersect.csv')

# N = 301
# name = []
# for i in range(N):
#     name.append('pre_'+ str(i))
# word_embed = pd.read_csv('../data/word_embed.csv',sep=' ',header=None,names=name)
# word_embed.drop('pre_0',axis=1, inplace=True)

train = pd.concat([
    train_difflib,
    # train_embedding,
    train_fuzz,
    train_len,
    train_match,
    train_word_match,
    train_match_2,
    train_simhash,
    train_ngram,
    # train_tfidf,
    train_word_ngram,

    train_graph_chars_pagerank,
    train_graph_pagerank,
    train_graph_chars_freq,
    train_graph_freq,
    # word_embed
    # train_graph_intersect,
], axis=1)

print(train.columns)

test_difflib = pd.read_csv(path + 'test_feature_difflib.csv')
test_word_difflib = pd.read_csv(path + 'test_feature_words_difflib.csv')
# test_embedding = pd.read_csv('test_feature_embedding.csv')
test_fuzz = pd.read_csv(path + 'test_feature_fuzz.csv')
test_len = pd.read_csv(path + 'test_feature_len.csv')
test_match = pd.read_csv(path + 'test_feature_match.csv')
test_word_match = pd.read_csv(path + 'test_feature_word_match.csv')
test_match_2 = pd.read_csv(path + 'test_feature_match_2.csv')
# test_oof = pd.read_csv('test_feature_oof.csv')
test_simhash = pd.read_csv(path + 'test_feature_simhash.csv')
test_tfidf = pd.read_csv(path + 'test_feature_tfidf.csv')
test_ngram = pd.read_csv(path + 'test_ngram_feature.csv')
test_word_ngram = pd.read_csv(path + 'test_word_ngram_feature.csv')

# test_graph_clique = pd.read_csv('test_feature_graph_clique.csv')
test_graph_chars_pagerank = pd.read_csv(path + 'test_feature_graph_chars_pagerank.csv')
test_graph_pagerank = pd.read_csv(path + 'test_feature_graph_pagerank.csv')
test_graph_chars_freq = pd.read_csv(path + 'test_feature_graph_chars_freq.csv')
test_graph_freq = pd.read_csv(path + 'test_feature_graph_freq.csv')
# test_graph_intersect = pd.read_csv(path + 'test_feature_graph_intersect.csv')

test = pd.concat([
    test_difflib,
    # test_embedding,
    test_fuzz,
    test_len,
    test_match,
    test_word_match,
    test_match_2,
    test_simhash,
    test_ngram,
    # test_tfidf,

    test_word_ngram,

    test_graph_chars_pagerank,
    test_graph_pagerank,
    test_graph_chars_freq,
    test_graph_freq,
    # test_graph_intersect,
], axis=1)
# print(test.columns)
# test.drop()

y = pd.read_csv('../data/train.csv')['label']

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.02,
    'min_child_weight': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.9,
    'max_depth': 8,
    'silent': 1,
    'seed': 1023,
}

n_fold = 5
kf = KFold(n=train.shape[0], n_folds=n_fold, shuffle=True, random_state=2017)

n = 0
for index_train, index_eval in kf:

    x_train, x_eval = train.iloc[index_train], train.iloc[index_eval]
    y_train, y_eval = y[index_train], y[index_eval]

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_eval, label=y_eval)
    watchlist = [(d_valid, 'valid')]

    bst = xgb.train(params, d_train, 40000, watchlist, early_stopping_rounds=100, verbose_eval=100)

    print('start predicting on test...')
    testpreds = bst.predict(xgb.DMatrix(test))
    if n > 0:
        totalpreds = totalpreds + testpreds
    else:
        totalpreds = testpreds

    bst.save_model('xgb_model_fold_{}.model'.format(n))
    n += 1

totalpreds = totalpreds / n

# submit result
sub = pd.DataFrame()
# sub['y_pre'] = list(totalpreds[:,0])
sub['y_pre'] = pd.Series(totalpreds)
sub.to_csv(path + 'xgb_prediction.csv',index=False)