# -*- coding: utf-8 -*-
# @Time    : 6/20/18 10:32 PM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_graph_pagerank_feature.py
import pandas as pd


question = pd.read_csv('../data/question.csv').fillna("")
question = question[['qid','chars']]

train = pd.read_csv('../data/train.csv').fillna("")
test = pd.read_csv('../data/test.csv').fillna("")
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','chars_x','chars_y']]
train.columns = ['label','chars1','chars2']
len_train = train.shape[0]

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['chars_x','chars_y']]
test.columns = ['chars1','chars2']


def generate_qid_graph_table(row):
    hash_key1 = row["chars1"]
    hash_key2 = row["chars2"]

    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)


qid_graph = {}
train.apply(generate_qid_graph_table, axis=1)
test.apply(generate_qid_graph_table, axis=1)


def pagerank():
    MAX_ITER = 40
    d = 0.85

    pagerank_dict = {i: 1 / len(qid_graph) for i in qid_graph}
    num_nodes = len(pagerank_dict)

    for iter in range(0, MAX_ITER):

        for node in qid_graph:
            local_pr = 0

            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor] / len(qid_graph[neighbor])

            pagerank_dict[node] = (1 - d) / num_nodes + d * local_pr

    return pagerank_dict


pagerank_dict = pagerank()


def get_pagerank_value(row):
    return pd.Series({
        "q1_pr": pagerank_dict[row["chars1"]],
        "q2_pr": pagerank_dict[row["chars2"]]
    })


pagerank_feats_train = train.apply(get_pagerank_value, axis=1)
pagerank_feats_test = test.apply(get_pagerank_value, axis=1)

path = '../data/pre_feature/'
pagerank_feats_train.to_csv(path + 'train_feature_graph_chars_pagerank.csv', index=False)
pagerank_feats_test.to_csv(path + 'test_feature_graph_chars_pagerank.csv', index=False)