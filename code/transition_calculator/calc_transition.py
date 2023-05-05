import pickle
from collections import defaultdict
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import sys

sys.path.append("..")
from toputils import *

N_PRED = 15 
PRIOR_W = 3

tgttop, _, config = read_config()
G_path = f"../data_interface/G_{tgttop}.pkl"
input_paths = [
        f"../data_interface/trajs_matched_{tgttop}_{r}.pkl" 
        for r in list(config["regions"].keys()) + ["all"]
    ]
ouput_cn2n2A = f"../data_interface/cn2n2A_{tgttop}.pkl"  # camera_node2node2A
ouput_cn2n2Ap = f"../data_interface/cn2n2Ap_{tgttop}.pkl"
ouput_n2A = f"../data_interface/n2A_{tgttop}.pkl"
ouput_n2Ap = f"../data_interface/n2Ap_{tgttop}.pkl"
output_h2transition = f"../data_interface/h2transition_{tgttop}.pkl"

G = pickle.load(open(G_path, "rb"))
edges = {i["id"]: i for i in G.edges.values()}
cams = {i for i, j in G.nodes.items() if j.get("camera", None)}

print("Loading matched traj...")
matched_traj = []
for input_path in input_paths:
    matched_traj += [
        [x[0] for x in tj["path"]]
        for tj in tqdm(pickle.load(open(input_path, "rb")))
    ]


def edges2nodes(arr):
    return [edges[i]["od"][0] for i in arr] + [edges[arr[-1]]["od"][1]]


def prior_statistic(matched_traj):
    """
    统计先验概率
    对于经过某个摄像头的所有轨迹, 取轨迹到达此摄像头前的K条路内的路径, 依频率估计转移概率
    输出为对每个node维护一个矩阵A, 行表示入路, 列表示出路, 行归一. A[i][j]表示从入路i去出路j的概率
    """
    # 统计以某摄像头所在node为终点的路径转移频数
    cam_to_pred_routes = defaultdict(list)  # 存在摄像头的node 的前序路径
    for route in tqdm(matched_traj):
        nodes = edges2nodes(route)
        t_dict = {}
        for i, node in enumerate(nodes):
            if i > 1:  # 摄像头之前至少有2条路才有意义
                if node in cams:
                    t_dict[node] = i  # 存在摄像头的node最后一次出现的位置
        for node, pos in t_dict.items():
            cam_to_pred_routes[node].append(route[:pos][-N_PRED:])
    cam_to_node_to_A = defaultdict(dict)  # 第一层key为存在摄像头的node 第二层key为node 值为A
    for cam, pred_routes in tqdm(cam_to_pred_routes.items()):
        for route in pred_routes:
            for rin, rout in zip(route, route[1:]):
                node = edges[rin]["od"][1]
                preds = G.nodes[node]["pred"]
                succs = G.nodes[node]["succ"]
                assert preds and succs, (preds, succs)
                if node not in cam_to_node_to_A[cam]:
                    cam_to_node_to_A[cam][node] = np.zeros(
                        (len(preds), len(succs)), dtype=float
                    )
                rin = preds.index(rin)
                rout = succs.index(rout)
                cam_to_node_to_A[cam][node][rin][rout] += 1
    # pickle.dump(cam_to_node_to_A, open(ouput_cn2n2A, "wb"))
    save_cam_to_node_to_A = deepcopy(cam_to_node_to_A)

    # 不管终点，直接统计转移频数
    node_to_A = {}
    for route in tqdm(matched_traj):
        if len(route) < 2:
            continue
        nodes = [edges[x]["od"][0] for x in route[1:]]
        for rin, rout, node in zip(route, route[1:], nodes):
            preds = G.nodes[node]["pred"]
            succs = G.nodes[node]["succ"]
            if node not in node_to_A:
                node_to_A[node] = np.zeros((len(preds), len(succs)), dtype=float)
            rin = preds.index(rin)
            rout = succs.index(rout)
            node_to_A[node][rin][rout] += 1
    # pickle.dump(node_to_A, open(ouput_n2A, "wb"))
    save_node_to_A = deepcopy(node_to_A)

    # 频率转换为概率，按行归一化  不管终点
    for A in node_to_A.values():
        if A.shape[1] == 1:
            A[:] = 1
        else:
            A += PRIOR_W + 1  # 使用更强一点的均匀先验
            A /= np.sum(A, axis=1).reshape(-1, 1)
    # pickle.dump(node_to_A, open(ouput_n2Ap, "wb"))

    # 频率转换为概率  终点为摄像头
    for n2a in cam_to_node_to_A.values():
        for node, A in n2a.items():
            rin, rout = A.shape
            if rout == 1:
                A[:] = 1
                continue
            B = A.copy()
            A += PRIOR_W
            A /= np.sum(A, axis=1).reshape(-1, 1)
            if np.sum(B) >= 4 * rout:
                t = np.sum(B, axis=0)  
                t += PRIOR_W * rin
                t /= np.sum(t)
                for i, (a, b) in enumerate(zip(A, B)):
                    sum_b = np.sum(b)
                    tmp = max(3, min(rout, 10))
                    if sum_b <= tmp:
                        tmp = sum_b / tmp
                        tmp = 0.3 + 0.3 * tmp
                        A[i] = a * tmp + t * (1 - tmp)

            if np.sum(B) < 1.5 * rout:
                A[:] = A * 0.6 + node_to_A[node] * 0.4
            assert np.allclose(np.sum(A, axis=1), 1)
    # pickle.dump(cam_to_node_to_A, open(ouput_cn2n2Ap, "wb"))
    return save_cam_to_node_to_A, cam_to_node_to_A, save_node_to_A, node_to_A


if __name__ == "__main__":
    a, b, c, d = prior_statistic(matched_traj)
    pickle.dump(a, open(ouput_cn2n2A, "wb"))
    pickle.dump(b, open(ouput_cn2n2Ap, "wb"))
    pickle.dump(c, open(ouput_n2A, "wb"))
    pickle.dump(d, open(ouput_n2Ap, "wb"))