# -*- coding: utf-8 -*-
# @Time    : 6/20/18 10:31 PM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : gen_graph_clique_feature.py
import networkx as nx
import pandas as pd
from itertools import combinations

question = pd.read_csv('../data/question.csv')
question = question[['qid','chars']]

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','chars_x','chars_y']]
train.columns = ['label','question1','question2']
len_train = train.shape[0]

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['chars_x','chars_y']]
test.columns = ['question1','question2']

df_feat = pd.DataFrame()

df = pd.concat([train[['question1', 'question2']], test[['question1', 'question2']]], axis=0)


G = nx.Graph()
edges = [tuple(x) for x in df[['question1', 'question2']].values]
G.add_edges_from(edges)

map_label = dict(((x[0], x[1])) for x in df[['question1', 'question2']].values)
map_clique_size = {}
cliques = sorted(list(nx.find_cliques(G)), key=lambda x: len(x))
for cli in cliques:
    for q1, q2 in combinations(cli, 2):
        if (q1, q2) in map_label:
            map_clique_size[q1, q2] = len(cli)
        elif (q2, q1) in map_label:
            map_clique_size[q2, q1] = len(cli)

df['clique_size'] = df.apply(lambda row: map_clique_size.get((row['question1'], row['question2']), -1), axis=1)


path = '../data/pre_feature/'
df[['clique_size']][:len_train].to_csv(path + 'train_feature_graph_clique.csv', index=False)
df[['clique_size']][len_train:].to_csv(path + 'test_feature_graph_clique.csv', index=False)
