# -*- coding: utf-8 -*-
# @Time    : 6/21/18 5:38 AM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_word_ngram_feature.py
import numpy as np
import pandas as pd
from collections import Counter
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

# df_feat = pd.DataFrame()
# df_data = pd.concat([train,test], axis=0)

def get_weight(count, eps=10000, min_count=2):
    # 单词权重定义，一个词的个数大于2，开始给予权重
    return 0 if count < min_count else 1 / (count + eps)

# 单词权重
train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
# Counter用来计数，https://blog.csdn.net/Shiroh_ms08/article/details/52653385
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


# 停用词english
stops = set(stopwords.words("english"))

def word_shares(row):
    q1_list = str(row['question1']).lower().split()
    q1 = set(q1_list)
    # python 并集union, 交集intersection, 差集difference，https://blog.csdn.net/lanyang123456/article/details/77596349
    q1words = q1.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0:0:0:0'

    q2_list = str(row['question2']).lower().split()
    q2 = set(q2_list)
    q2words = q2.difference(stops)
    if len(q2words) == 0:
        return '0:0:0:0:0:0:0:0'

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q2_weights

    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)) #count share
    R31 = len(q1stops) / len(q1words) #stops in q1
    R32 = len(q2stops) / len(q2words) #stops in q2
    Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))
    Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)

df = pd.concat([train, test])
df['word_shares'] = df.apply(word_shares, axis=1, raw=True)


df_feature = pd.DataFrame()

df_feature['word_match_ratio_word_2']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
df_feature['word_match_ratio_word_2_root'] = np.sqrt(df_feature['word_match_ratio_word_2'])
df_feature['tfidf_word_match_word_ratio_2'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
df_feature['shared_word_count_2']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

df_feature['stops1_word_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
df_feature['stops2_word_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
df_feature['shared_word_2gram']     = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
df_feature['word_match_word_cosine']= df['word_shares'].apply(lambda x: float(x.split(':')[6]))
df_feature['words_word_hamming']    = df['word_shares'].apply(lambda x: float(x.split(':')[7]))
df_feature['diff_stops_word_r']     = df_feature['stops1_word_ratio'] - df_feature['stops2_word_ratio']


path = '../data/pre_feature/'
df_feature[:train.shape[0]].to_csv(path + 'train_word_ngram_feature.csv', index=False)
df_feature[train.shape[0]:].to_csv(path + 'test_word_ngram_feature.csv', index=False)