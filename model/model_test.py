# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_test
   Description :
   Author :       LiYang
   date：          7/6/18
-------------------------------------------------
   Change Activity:
                   7/6/18:
-------------------------------------------------
"""
__author__ = 'LiYang'

import matplotlib.pyplot as plt
import pandas as pd
import h5py  #导入工具包

# loss = pd.read_hdf('./decomposable_attention_5fold_0curfold.h5')
f = h5py.File('decomposable_attention_5fold_0curfold.h5', 'r')
# print(f.keys())
# [u'batch_normalization_1', u'batch_normalization_2', u'embed', u'global_average_pooling1d_1', u'global_average_pooling1d_2', u'global_max_pooling1d_1', u'global_max_pooling1d_2', u'lambda_1', u'merge_1', u'merge_10', u'merge_2', u'merge_3', u'merge_4', u'merge_5', u'merge_6', u'merge_7', u'merge_8', u'merge_9', u'sequential_3', u'spatial_dropout1d_1', u'spatial_dropout1d_2', u'time_distributed_1', u'time_distributed_2', u'time_distributed_3', u'tune', u'words1', u'words2']

# a = f['embed'][:]                    #取出主键为data的所有的键值
print(f.values())
f.close()