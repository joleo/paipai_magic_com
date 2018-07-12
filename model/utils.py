# -*- coding: utf-8 -*-
# @Time    : 7/1/18 3:35 AM
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : utils.py

import pickle

def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

class SaveData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)