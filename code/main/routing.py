"""
先验/似然/后验概率计算, 最大后验路径求解
"""
import os
import pickle
import sys
from collections import defaultdict
from itertools import islice
from math import exp, inf

import matplotlib.pyplot as plt
import numpy as np
from eviltransform import distance
from matplotlib.ticker import MultipleLocator
from networkx import shortest_simple_paths
from tqdm import tqdm

sys.path.append("..")
from toputils import *

K = 10
SIGMA_RATIO = 0.8  # sigma是mu的比例
PRIOR_W = 5  # 均匀先验的强度
MAX_T_STAY = 600


def gauss(v, mu):
    sigma = mu * SIGMA_RATIO
    return exp(-((v - mu) ** 2) / sigma**2 / 2)


class Router:
    def __init__(self, target=None):
        if target is None:
            tgttop, target = read_config()[:2]
        else:
            tgttop, _, config = read_config()
            
        G_path = f"../data_interface/G_{tgttop}.pkl"
        stpth_path = f"../data_interface/shortest_path_{tgttop}_{target}.pkl"
        input_speed = f'../data_interface/road_speed_{tgttop}.pkl'
        input_cn2n2A = f"../data_interface/cn2n2A_{tgttop}.pkl"
        input_cn2n2Ap = f"../data_interface/cn2n2Ap_{tgttop}.pkl"
        input_n2A = f"../data_interface/n2A_{tgttop}.pkl"
        input_n2Ap = f"../data_interface/n2Ap_{tgttop}.pkl"

        self.G = pickle.load(open(G_path, "rb"))
        self.edge_info_dict = {e["id"]: [u, v, e] for u, v, e in self.G.edges(data=True)}

        # 每时间片的道路速度期望
        self.speed_dicts = pickle.load(open(input_speed, "rb"))

        # 先验转移频数、概率
        cn2n2A = pickle.load(open(input_cn2n2A, "rb"))
        cn2n2Ap = pickle.load(open(input_cn2n2Ap, "rb"))
        n2A = pickle.load(open(input_n2A, "rb"))
        n2Ap = pickle.load(open(input_n2Ap, "rb"))
        self.transition = [cn2n2A, cn2n2Ap, n2A, n2Ap]

        # 预计算 某edge作为node的前驱时是第几条, 作为node的后继时是第几条
        self.edge_to_pred_index = {}
        self.edge_to_succ_index = {}
        for node in self.G.nodes.values():
            for i, edge in enumerate(node["pred"]):
                self.edge_to_pred_index[edge] = i
            for i, edge in enumerate(node["succ"]):
                self.edge_to_succ_index[edge] = i

        # 预计算好的, 所有摄像头node间的, 10_shortest_paths
        if target != "all":
            self.shortest_path_results = pickle.load(open(stpth_path, 'rb'))
        else:
            self.shortest_path_results = {}
            for stpth_path in [f"../data_interface/shortest_path_{tgttop}_{r}.pkl" for r in config["regions"]]:
                self.shortest_path_results.update(pickle.load(open(stpth_path, 'rb')))

    def route_prior(self, route, return_p_nostart=False, h=None):
        """
        计算某route的先验概率
        route的起点,终点应为某个摄像头
        """
        if h is not None:
            cn2n2A, cn2n2Ap, n2A, n2Ap = self.h2transition[h]
        else:
            cn2n2A, cn2n2Ap, n2A, n2Ap = self.transition
        u = self.edge_info_dict[route[0]][0]
        v = self.edge_info_dict[route[-1]][1]
        A_dict = cn2n2A.get(v, n2A)
        A_dict_p = cn2n2Ap.get(v, n2Ap)
        # 第一条路的概率
        if u in A_dict:
            tmp = np.sum(A_dict[u], axis=0)
            tmp += PRIOR_W * A_dict[u].shape[0]
            tmp /= np.sum(tmp)
            p_start = tmp[self.edge_to_succ_index[route[0]]]
        else:
            p_start = 1 / len(self.G.nodes[u]["succ"])
        # 后续的转移概率
        p_nostart = 1.0
        nodes = [self.edge_info_dict[x][0] for x in route[1:]]
        for rin, rout, node in zip(route, route[1:], nodes):
            A = A_dict_p.get(node, n2Ap.get(node, None))
            if A is None:
                p_nostart *= 1 / len(self.G.nodes[node]["succ"])
            else:
                p_nostart *= A[self.edge_to_pred_index[rin]][self.edge_to_succ_index[rout]]
        if return_p_nostart:
            return p_start * p_nostart, p_nostart
        else:
            return p_start * p_nostart
    
    def my_k_shortest_paths(self, u, v, k):
        paths_gen = shortest_simple_paths(self.G, u, v, 'length')
        for path in islice(paths_gen, 0, k):
            yield path

    def route_likelihood(self, route, ttm, slot, route_type="node"):
        """
        计算某route的似然概率
        route可以为edge序列或node序列, 由route_type指定
        """
        speed_dict = self.speed_dicts[slot]
        total_etm = 0
        if route_type == "node":
            edges = []
            for n1, n2 in zip(route, route[1:]):
                edge = self.G.edges[n1, n2]
                length = edge["length"]
                edge_id = edge["id"]
                speed = speed_dict[edge_id]
                etm = length / speed
                total_etm += etm
                edges.append(edge_id)
            
            if etm < 0.1:  # 处理length=0, etm=0的路
                for n1, n2 in zip(route, route[1:]):
                    edge = self.G.edges[n1, n2]
                    length = edge["length"]
                    speed = speed_dict[edge["id"]]
                    etm = length / speed
                    if etm > 0.1:
                        break

            v = length / (ttm * etm / total_etm)
            return gauss(v, speed), edges  # , total_etm
        elif route_type == "edge":
            for edge in route:
                length = self.edge_info_dict[edge][-1]["length"]
                speed = speed_dict[edge]
                etm = length / speed
                total_etm += etm
            
            if etm < 0.1:  # 处理length=0, etm=0的路
                for edge in route:
                    length = self.edge_info_dict[edge][-1]["length"]
                    speed = speed_dict[edge]
                    etm = length / speed
                    if etm > 0.1:
                        break

            v = length / (ttm * etm / total_etm)
            return gauss(v, speed)  # , total_etm

    def p_stay(self, t):
        """若两相邻摄像头在同一路口, 则按此计算'路径'概率"""
        return max(0, 1 - t / MAX_T_STAY)

    def MAP_routing(self, u, v, ut, vt, k=K, return_route=False):
        """
        给定u, v, ut, vt，求最大后验路径
        u, v应是存在摄像头的node
        当恢复经过多个摄像头的路径时, 
            应在每两个相邻摄像头间用此函数分别恢复路径, 
            最终路径为这些分段路径的拼接, 概率取几何平均
        """
        ttm = vt - ut
        if ttm == 0:
            ttm += 0.01
        slot = int(ut / 3600)
        assert ttm > 0
        if u != v:  # k_shortest_paths无法支持u=v的情况，将返回[[u]]
            try:    # 可能有少数摄像头间不连通
                proposals = list(islice(self.shortest_path_results[u, v], k))
            except:
                try:
                    proposals = list(self.my_k_shortest_paths(u, v, k))  # 支持跨区域导航(预计算中只在每个区域内算了)
                except:
                    proposals = []
        else:
            proposals = [[u]]  # 原地停留
            for inter in self.G[u].keys():
                # 手动往外走一步作为中介点，再走回来
                # 此处不是在摄像头间寻路，无法直接读结果
                try:
                    proposals.extend([u] + t for t in self.my_k_shortest_paths(inter, v, 5))
                except:
                    pass
        if len(proposals) == 0:
            if return_route:
                return [], 1e-12
            return 1e-12

        posteriors = []
        for nodes in proposals:
            if len(nodes) > 1:
                likelihoood, route = self.route_likelihood(nodes, ttm, slot)
                prior = self.route_prior(route)
                posteriors.append(likelihoood * prior)
            else:
                assert len(nodes) == 1
                posteriors.append(self.p_stay(ttm))
        if return_route:
            r, p = max(zip(proposals, posteriors), key=lambda x: x[1])
            # 返回的是除u,v外的中间node
            return r[1:-1], max(p, 1e-12)  # 若为原地停留, 中间node为[]
        return max(max(posteriors), 1e-12)


def validate(target, mode="gaode"):
    """
    用高德轨迹的路网匹配结果, 计算其概率得分, 
    画似然、先验概率直方图, 验证概率模型的可靠性
    """
    tgttop = read_config()[0]
    if mode == "gt":
        path = f"data/routes_between_cameras_{tgttop}_{target}_gt.pkl"
    else:
        path = f"data/routes_between_cameras_{tgttop}_{target}.pkl"
    if os.path.exists(path):
        routes_between_cameras = pickle.load(open(path, "rb"))
    else:
        G = pickle.load(open(f"../data_interface/G_{tgttop}.pkl", "rb"))
        eid2e = {e["id"]: e for u, v, e in G.edges(data=True)}

        if mode == "gaode":
            trajs_orig = pickle.load(open(f'../map_matcher/data/trajs_{tgttop}_{target}.pkl', 'rb'))
            print("trajs_orig:", len(trajs_orig))
            trajs_matched = pickle.load(open(f"../data_interface/trajs_matched_{tgttop}_{target}.pkl", 'rb'))
            print("trajs_matched:", len(trajs_matched))
            trajs_orig = [trajs_orig[x["index"]] for x in trajs_matched]
        else:
            vid2trajs_orig = pickle.load(open(f"../map_matcher/data/gt_vid2trajs_{tgttop}.pkl", "rb"))
            vid2trajs_matched = pickle.load(open(f"../data_interface/gt_vid2trajs_matched_{tgttop}.pkl", "rb"))
            trajs_matched = []
            for vid, tjs in vid2trajs_matched.items():
                for tj in tjs:
                    tj["vid"] = vid
                    trajs_matched.append(tj)
            trajs_orig = []
            for vid, tjs in vid2trajs_matched.items():
                tjs_orig = vid2trajs_orig[vid]
                for tj in tjs:
                    for e in tj["path"]:
                        if e[1]:
                            t = e[1][0]["point"][-1]
                            break
                    else:
                        assert False
                    for tj2 in tjs_orig:
                        if tj2[0][-1] <= t <= tj2[-1][-1]:
                            break
                    else:
                        assert False
                    trajs_orig.append(tj2)
            assert len(trajs_matched) == len(trajs_orig)
            print("trajs_matched:", len(trajs_matched))

        routes_between_cameras = defaultdict(list)
        for item, traj_orig in zip(tqdm(trajs_matched), trajs_orig):
            vid = item["index" if mode == "gaode" else "vid"]
            # 找匹配轨迹经过的cnid, 及前后最近匹配点的时间
            nodes_with_camera = []
            edges = item['path']
            len_edges = len(edges)
            last_v = eid2e[edges[0][0]]["od"][0]
            for i, edge in enumerate(edges):    # edge[1]: [{'order', 'point':[lon, lat, tm], 'orig_point'}]
                edge_id = edge[0]
                u, v = eid2e[edge_id]["od"]
                assert u == last_v
                last_v = v
                if i != 0:  # 第一条边的起点认为没有被经过
                    if G.nodes[u].get('camera', None):  # 起点是camera
                        for j in range(i, len_edges):
                            if edges[j][1]:
                                end_time = edges[j][1][0]['point'][-1]     # 找到经过该camera之后的第一个点的时间
                                break
                        assert nodes_with_camera[-1]['node_id'] == u
                        nodes_with_camera[-1]['end_time'] = end_time
                if i != len_edges - 1:          # 最后一条边的终点认为没有被经过
                    if G.nodes[v].get('camera', None):  # 终点是camera
                        for j in range(i, -1, -1):
                            if edges[j][1]:
                                start_time = edges[j][1][-1]['point'][-1]  # 找到经过该camera之前的第一个点的时间
                                break
                        nodes_with_camera.append({'node_id': v, 'on_which_edge': i+1, 'start_time': start_time})  # on_which_edge: 在哪条边的起点
            if len(nodes_with_camera) < 2:  # 感兴趣至少经过两个camera的轨迹
                continue
            # 取原始轨迹中时间范围在lb和up间的点，找到与node空间最接近的点，作为通过camera的时间
            last_ub = -inf
            last_add = 0
            search_index = 0
            for camera in nodes_with_camera:
                node = G.nodes[camera['node_id']]
                lon, lat = node['x'], node['y']
                lb = camera['start_time']
                ub = camera['end_time']
                if lb < last_ub:
                    search_index -= last_add
                last_ub = ub
                assert traj_orig[search_index][-1] <= lb
                points = []                    
                for i, p in enumerate(traj_orig[search_index:]):
                    t = p[-1]
                    if t >= lb:
                        if t <= ub:
                            points.append(p)
                        else:
                            break
                last_add = max(i - 1, 0)
                search_index += last_add
                assert points
                diss = [distance(lat, lon, p[1], p[0]) for p in points]
                min_dis = min(diss)
                camera['true_time'] = points[diss.index(min_dis)][-1]
                camera['true_time_error'] = min_dis

            nodes_with_camera = [x for x in nodes_with_camera if x['true_time_error'] < 100]
            if len(nodes_with_camera) < 2:
                continue

            for c1, c2 in zip(nodes_with_camera, nodes_with_camera[1:]):
                route = [x[0] for x in edges[c1['on_which_edge']:c2['on_which_edge']]]
                start_time = c1['true_time']
                end_time = c2['true_time']
                if end_time > start_time:
                    ave_speed = sum([eid2e[x]['length'] for x in route]) / (end_time - start_time)
                    if 1 < ave_speed < 30:
                        routes_between_cameras[vid].append({
                            'start_node': c1['node_id'], 
                            'end_node': c2['node_id'], 
                            'route': route, 'start_time': start_time, 'end_time': end_time})
        pickle.dump(routes_between_cameras, open(path, "wb"))
    print("routes_between_cameras:", len(routes_between_cameras))

    # 画直方图
    router = Router(target)
    lis = []
    pis = []
    pos = []
    for routes in tqdm(routes_between_cameras.values()):
        for item in routes:
            u, v, true_route, ut, vt = item.values()
            li = router.route_likelihood(true_route, vt - ut, int(ut/3600), "edge")
            lis.append(li)
            pi = router.route_prior(true_route)
            pis.append(pi)
            pos.append(li * pi)
    plt.figure(figsize=(10, 15))
    groups = np.arange(0, 1.05, 0.05)
    plt.subplot(3, 1, 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.hist(lis, groups, histtype='step', rwidth=0.8, density=True, cumulative=True)
    plt.title('likelihood')
    plt.subplot(3, 1, 2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.hist(pis, groups, histtype='step', rwidth=0.8, density=True, cumulative=True)
    plt.title('prior')
    plt.subplot(3, 1, 3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.hist(pos, groups, histtype='step', rwidth=0.8, density=True, cumulative=True)
    plt.title('post')
    if mode == "gt":
        plt.savefig(f'figure/routing_validate_{tgttop}_gt.png')
    else:
        plt.savefig(f'figure/routing_validate_{tgttop}_{target}.png')


def main():
    targets = [
        "cluster1",
        "cluster2",
        "cluster3",
        "cluster4",
        "cluster5",
    ]
    for target in targets:
        validate(target, mode="gaode")
    validate("all", mode="gt")


if __name__ == "__main__":
    main()
