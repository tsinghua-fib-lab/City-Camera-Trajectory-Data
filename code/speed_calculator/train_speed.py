"""
矩阵分解补全道路速度估计
"""
import json
import os
import pickle
import random
import sys
from math import sqrt

import numpy as np
import requests
import torch
import tqdm
from calc_speed import get_level2speed
from coord_convert.transform import gcj2wgs, wgs2gcj
from tqdm import tqdm

sys.path.append("..")
from toputils import *

tgttop, _, config = read_config()
G_path = f"../data_interface/G_{tgttop}.pkl"
input_full_simple = f"../data_interface/road_speed_simple_full_{tgttop}.pkl"
input_slice = f"../data_interface/road_speed_slice_{tgttop}.pkl"
input_slice_simple = f"../data_interface/road_speed_slice_simple_{tgttop}.pkl"
output_path = f'../data_interface/road_speed_{tgttop}.pkl'

G = pickle.load(open(G_path, "rb"))
edges = list(G.edges(data=True))
# lengths = [x[2]['length'] for x in edges]
# print(np.mean(lengths))  # 平均道路长度183.9m

N_edge = len(edges)
N_tm = 24

eid2s_full_complement = pickle.load(open(input_full_simple, "rb"))
eid2level = {e["id"]: e["highway"] for u, v, e in G.edges(data=True)}
level2s_full = get_level2speed(eid2level, eid2s_full_complement)
t = sorted(list(level2s_full.items()), key=lambda x:-x[1])
ranked_levels = [x[0] for x in t]
level_effect = [x[1] for x in t]     # 初始化参数, 道路等级对速度的效应: 每种等级道路的平均速度
N_feature = len(ranked_levels) + 1   # 显式特征维度
N_latent = 50                        # 隐变量空间维数
poi_effect = -0.01                   # 初始化参数, 道路附近POI数量对速度的效应

def train_data_matrix(speed_dicts):
    """
    X_truth: 各道路在各时间片的速度估计(有缺失)
    M: X_truth是否缺失的0,1指示矩阵
    """
    M = np.zeros((N_edge, N_tm), dtype=int)
    X_truth = np.zeros((N_edge, N_tm), dtype=float)
    for j in range(N_tm):
        speed_dict = speed_dicts[j]
        for i, edge in enumerate(edges):
            id = edge[2]['id']
            if id in speed_dict:
                X_truth[i][j] = speed_dict[id]
                M[i][j] = 1
    return X_truth, M


def road_feature_matrix(cache_poi_num):
    """
    F: 道路显式特征矩阵: 周边POI数量, 道路等级(one-hot)
    """
    def level_rank(level, ranked_levels):
        if isinstance(level, str):
            rank = [ranked_levels.index(level)]
        else:
            rank = [ranked_levels.index(t) for t in level]
        return rank

    F = np.zeros((N_edge, N_feature), dtype=float)
    for i, edge in tqdm(enumerate(tqdm(edges))):
        info = edge[2]
        rank = level_rank(info['highway'], ranked_levels)
        for r in rank:
            F[i][1+r] = 1 / len(rank)  # one-hot编码表示道路level，若道路level为list(有多种)，在各不同种之间平分

    return F, cache_poi_num


def matrix_factorization(X_truth, M, F, device=torch.device('cuda'), train_portion=1, N_iter=15000):
    """
    矩阵分解补充缺失值
    X_rebuild = X_static + X_dynamic = dot(F,W) + dot(U,V.T)
    J = 1/2 * norm2[ M * (X_truth - FW - UV.T) ] + 1/2 * lambda * [ norm2(W) + norm2(U) + norm2(V) ]
    """
    random.seed(233)
    np.random.seed(233)
    torch.manual_seed(233)
    torch.cuda.manual_seed(233)
    
    t = np.where(M==1)
    t = list(zip(*t))
    index_train = random.sample(t, round(train_portion*len(t)))
    M_train = np.zeros((N_edge, N_tm), dtype=int)
    M_valid = np.zeros((N_edge, N_tm), dtype=int)
    for i, j in index_train:
        M_train[i][j] = 1
    a = M==1
    b = M_train==0
    c = a*b
    M_valid[c] = 1
    assert (M_train + M_valid == M).all()

    X_truth = torch.tensor(X_truth, dtype=torch.float64, device=device)
    # M = torch.tensor(M, dtype=torch.bool, device=device)
    M_train = torch.tensor(M_train, dtype=torch.bool, device=device)
    M_valid = torch.tensor(M_valid, dtype=torch.bool, device=device)
    F = torch.tensor(F, dtype=torch.float64, device=device)
    W = torch.tensor([poi_effect] + level_effect, dtype=torch.float64, device=device, requires_grad=True)
    U = torch.rand((N_edge, N_latent), device=device, requires_grad=True)
    V = torch.rand((N_tm, N_latent), device=device, requires_grad=True)
    print('F', F.shape, 'W', W.shape, 'U', U.shape, 'V', V.shape)
    
    opt = torch.optim.Adam([W, U, V], weight_decay=1e-4)  # L2正则，weight_decay
    best_iter = 0
    best_t2 = 10000
    with tqdm(range(N_iter)) as tq:
        for i in tq:
            X_rebuild = (F @ W).view(-1, 1) + U @ V.T
            loss = torch.mean(torch.square((X_rebuild - X_truth)[M_train]))
            opt.zero_grad()
            loss.backward()
            opt.step()

            t = loss.cpu().item()**0.5
            t2 = torch.mean(torch.square((X_rebuild - X_truth)[M_valid])).cpu().item()**0.5
            tq.set_description(f'loss: {t:.6f}, valid loss: {t2:.6f}')

            if train_portion < 1 and t2 < best_t2:
                best_t2 = t2
                best_iter = i
    if train_portion < 1:
        print("best iter:", best_iter, best_t2)
        return best_iter
    else:
        X_rebuild = X_rebuild.cpu().detach().numpy()
        return X_rebuild


if __name__ == '__main__':
    cache_path = f"data/X_truth_M_{tgttop}.npz"
    if os.path.exists(cache_path):
         t = np.load(cache_path)
         X_truth, M = t["X_truth"], t["M"]
    else:
        eid2s_each_h = pickle.load(open(input_slice, 'rb'))
        X_truth, M = train_data_matrix(eid2s_each_h)  # 道路 * 时间片 的速度真值矩阵, 真值mask矩阵
        np.savez(file=cache_path, X_truth=X_truth, M=M)
    print("X_truth:", X_truth.shape)
    print("M", M.shape)

    cache_path = f"data/F_{tgttop}.npy"
    if os.path.exists(cache_path):
        F = np.load(cache_path)
    else:
        cache_poi_path = f"data/poi_num_{tgttop}.pkl"
        if os.path.exists(cache_poi_path):
            cache_poi_num = pickle.load(open(cache_poi_path, "rb"))
            print(len(cache_poi_num))
        else:
            cache_poi_num = {}
        F, cache_poi_num = road_feature_matrix(cache_poi_num)
        pickle.dump(cache_poi_num, open(cache_poi_path, "wb"))
        np.save(cache_path, F)

    cache_path = f"data/X_rebuild_{tgttop}.npy"
    if os.path.exists(cache_path):
        X_rebuild = np.load(cache_path)
    else:
        best_iter = matrix_factorization(X_truth, M, F, train_portion=0.7)  # 先划分train/valid找到最佳训练次数
        X_rebuild = matrix_factorization(X_truth, M, F, N_iter=best_iter)   # 再全部train得到结果
        np.save(cache_path, X_rebuild)

    # 评价对X_truth的拟合效果
    X_delta = M * (X_rebuild - X_truth)
    print('MAE:', np.sum(np.abs(X_delta)) / np.sum(M))
    print('RMSE:', sqrt(np.sum(X_delta * X_delta) / np.sum(M)))

    # 评价对缺失值的预测效果
    # 将 矩阵分解结果 与 用简单规则分时间片补全的结果 对比
    X_simple = np.zeros((N_edge, N_tm), dtype=float)
    speed_dicts = pickle.load(open(input_slice_simple, 'rb'))
    for j in range(N_tm):
        speed_dict = speed_dicts[j]
        for i, edge in enumerate(edges):
            X_simple[i][j] = speed_dict[edge[2]['id']]
    X_delta = (1 - M) * (X_rebuild - X_simple)
    print('MAE:', np.sum(np.abs(X_delta)) / np.sum(1 - M))  # 2.86
    print('RMSE:', sqrt(np.sum(np.square(X_delta)) / np.sum(1 - M)))  # 3.59

    # 形成最终结果
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    X_result = np.zeros((N_edge, N_tm), dtype=float)
    for i in range(N_edge):
        for j in range(N_tm):
            if M[i][j] == 1:
                X_result[i][j] = X_truth[i][j]
            else:
                s1 = X_rebuild[i][j]                  # 矩阵分解补全结果
                s2 = X_simple[i][j]                   # 简单规则补全结果
                s3 = np.sum(F[i][1:] * level_effect)  # 道路等级平均结果
                # print(f'{s1:.2f}, {s2:.2f}, {s3:.2f}')

                if 0.75*s3 < s2 < 1.25*s3:
                    s_ref = s2
                elif 0.5*s3 < s2 < 1.5*s3:
                    s_ref = (s2 + s3) / 2
                else:
                    s_ref = s3

                if 0.75*s_ref < s1 < 1.25*s_ref:
                    s_result = s1
                    cnt1 += 1
                elif 0.5*s_ref < s1 < 1.5*s_ref:
                    s_result = (s1 + s_ref) / 2
                    cnt2 += 1
                else:
                    s_result = s_ref
                    cnt3 += 1
                X_result[i][j] = s_result

    print('adopt num:', cnt1)      # 15364
    print('weak-adopt num:', cnt2) # 3456
    print('reject num:', cnt3)     # 677
    
    speed_result_for_each_slice = [{edge[2]['id']: 0 for edge in edges} for i in range(N_tm)]
    for t in range(N_tm):
        speed_dict = speed_result_for_each_slice[t]
        for edge, speed in zip(speed_dict.keys(), X_result[:, t]):
            speed_dict[edge] = speed
    pickle.dump(speed_result_for_each_slice, open(output_path, 'wb'))
