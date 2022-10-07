'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
#from time import time

# from numba import jit, autojit
def getNovel(ranklist, cold_items, tail_items):
    count = 0
    for item in ranklist:
        if (item in cold_items) or (item in tail_items):
            count += 1
    return count/len(ranklist)

def getCC(cate_num, ranklist, gtItems, item_cate_dict): #类似于预测category的recall
    item_cate_arr = np.array(list(item_cate_dict.values()))
    rank_catelist = set(item_cate_arr[ranklist])
    gt_catelist = set(item_cate_arr[list(gtItems)])
    result = len(rank_catelist & gt_catelist) / len(gt_catelist)
    return result

def getILD(ranklist, item_cate_dict):
    K = len(ranklist)
    item_cate_arr = np.array(list(item_cate_dict.values()))
    tmp = 0
    catelist = item_cate_arr[ranklist]
    for i in range(len(catelist)):
        for j in range(i, len(catelist)):
            if catelist[i] != catelist[j]:
                tmp += 1   #两个item的category vecto计算cos距离，相当于相同category的加上1
    tmp = 2*tmp/(K*(K-1))
    return tmp

def getP(ranklist, gtItems):
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
    return p * 1.0 / len(ranklist)

def getR(ranklist, gtItems):
    r = 0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)

def getHitRatio(ranklist, gtItem):  #推荐一个的hit rate, @K中的命中数/len(gold test)，好像等同于recall
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return  dcg

def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg

def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg

def dcg_at_k(r, k, method=0):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def calculate_ndcg(groundtruth, rec_list, k=20):
    relevance_scores = np.zeros(len(rec_list)).tolist()

    for i, item in enumerate(rec_list):
        if item in groundtruth:
            relevance_scores[i] = 1

    score = ndcg_at_k(relevance_scores, k, 1)
    return score