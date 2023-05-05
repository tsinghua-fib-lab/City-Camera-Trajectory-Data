import json
import logging
import os
import pickle
import random
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
from math import ceil, sqrt

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import yaml
from eval import evaluate
from pmap import process_map
from routing import Router
from setproctitle import setproctitle
from sig_cluster import FlatSearcher, SigCluster
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map as tqdm_process_map
from tuner import Table
from utils import ljust, mean
import sys

sys.path.append("..")
from toputils import *

coloredlogs.install(fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s")

TYPE_ORDINARY_NOISE = 1
TYPE_STRONG_NOISE = 2
TYPE_ONE_NOISE = 3
TYPE_TWO_NOISE = 4
TYPE_LONG_NOISE = 5
TYPE_BLACK_LIST_NOISE = 6
TYPE_OUT_OF_SUBSET_NOISE = 7
TYPE_SINGLE_CLUSTER = 8

def length_clamp(x):
    l = np.linalg.norm(x)
    return x if l < MAX_LENGTH else x / l * MAX_LENGTH


def subsets(arr, k=0, max_return=1000):
    # 按元素个数从多到少返回arr的所有大小不小于k的非空子集, 且最多返回max_return个
    cnt = 0
    if cnt >= max_return:
        return
    for i in range(len(arr), max(0, k - 1), -1):
        for j in combinations(arr, i):
            yield j
            cnt += 1
            if cnt >= max_return:
                return


def merge_tm_adj_points(points, adj_range):
    """
    合并同一node处多个tm相接近的点(车经过摄像头的短时间内被拍摄多次)
    points: [(camera_node_id, tm, record_id)]
    adj_range：合并的时间差
    """
    # group by node
    node_to_tms = defaultdict(list)
    if isinstance(points[0][-1], list):
        for node, tm, i in points:
            node_to_tms[node].append((tm, i))
    else:
        for node, tm, i in points:
            node_to_tms[node].append(
                (tm, [i])
            )  # 会连续调用该函数两次, 为保证输入输出格式一致, 将record_id统一为list
    # merge by time gap
    merge_points = []
    for node, tms in node_to_tms.items():
        if len(tms) == 1:
            merge_points.append((node, tms[0][0], tms[0][1]))
        else:
            tms.sort(key=lambda x: x[0])
            min_tm = tms[0][0]
            one_cluster = [tms[0]]
            for tm, i in tms[1:]:
                if tm - min_tm <= adj_range:
                    one_cluster.append((tm, i))
                else:
                    a, b = list(zip(*one_cluster))
                    merge_points.append(
                        (node, np.mean(a), sum(b, []))
                    )  # 融合成一个点，时间取平均，索引合并
                    one_cluster = [(tm, i)]
                    min_tm = tm
            a, b = list(zip(*one_cluster))
            merge_points.append((node, np.mean(a), sum(b, [])))
    return merge_points


def cut_distant_points(points, tm_gap_gate):
    """按照时间间隔将轨迹切分成多段"""
    cut_points = []
    one_cut = [points[0]]
    tm_last = points[0][1]
    for point in points[1:]:
        tm = point[1]
        if tm - tm_last > tm_gap_gate:
            cut_points.append(one_cut)
            one_cut = [point]
        else:
            one_cut.append(point)
        tm_last = tm
    cut_points.append(one_cut)
    return cut_points


def detect_many_noise(cuts):
    """
    噪声检测的实现
    输入经过merge_tm_adj_points+cut_distant_points预处理后的轨迹
    输出噪声, 顺便抛出一些"召回请求"(最优子集对应恢复的轨迹中途经过了摄像头)
    """
    noises = []
    recall_attempts = []
    long_cuts = []
    total_len = sum([len(cut) for cut in cuts])
    for i, one_cut in enumerate(cuts):
        if len(one_cut) == 1:
            if (
                total_len > TYPE_ONE_NOISE_KEEP_TOTAL_LEN
            ):
                c, ct = one_cut[0][:2]
                flag = True
                if i > 0:
                    u, ut = cuts[i - 1][-1][:2]
                    if router.MAP_routing(u, c, ut, ct) > TYPE_ONE_NOISE_KEEP_GATE:
                        flag = False
                if flag and i < len(cuts) - 1:
                    v, vt = cuts[i + 1][0][:2]
                    if router.MAP_routing(c, v, ct, vt) > TYPE_ONE_NOISE_KEEP_GATE:
                        flag = False
                if flag:
                    noises.append((one_cut[0][-1], TYPE_ONE_NOISE))
        elif len(one_cut) == 2:
            (u, ut, _), (v, vt, _) = one_cut
            p = router.MAP_routing(u, v, ut, vt)
            if p < TYPE_TWO_NOISE_GATE:
                flag = True
                if i > 0:
                    prev_u, prev_ut = cuts[i - 1][-1][:2]
                    if (
                        router.MAP_routing(prev_u, u, prev_ut, ut)
                        > TYPE_TWO_NOISE_KEEP_GATE
                    ):
                        flag = False
                if flag and i < len(cuts) - 1:
                    succ_v, succ_vt = cuts[i + 1][0][:2]
                    if (
                        router.MAP_routing(v, succ_v, vt, succ_vt)
                        > TYPE_TWO_NOISE_KEEP_GATE
                    ):
                        flag = False
                if flag:
                    noises += [(x[-1], TYPE_TWO_NOISE) for x in one_cut]
            elif DO_RECALL_ATTEMPT:
                inter_nodes, _ = router.MAP_routing(u, v, ut, vt, return_route=True)
                inter_camera_nodes = [
                    node for node in inter_nodes if "camera" in G.nodes[node]
                ]
                if inter_camera_nodes:
                    recall_attempts.append(
                        ([u, ut], [v, vt], inter_camera_nodes)
                    )
        else:
            long_cuts.append((i, one_cut))

    for ii, one_cut in long_cuts:
        len_cut = len(one_cut)
        p_dict = {}  # 算过的概率不用再次计算
        sub_ps_raw = []
        sub_ps = []
        sub_idxs = []
        for sub_idx in subsets(list(range(len_cut)), k=ceil(len_cut / 2)):
            ps = []
            for i, j in zip(sub_idx, sub_idx[1:]):
                p = p_dict.get((i, j), None)
                if p is None:
                    u, ut = one_cut[i][:2]
                    v, vt = one_cut[j][:2]
                    p = router.MAP_routing(u, v, ut, vt)
                    p_dict[i, j] = p
                ps.append(p)
            p = np.exp(np.mean(np.log(ps)))
            sub_ps_raw.append(p)
            sub_ps.append(p * len(sub_idx) / len_cut)
            sub_idxs.append(sub_idx)

        max_sub_p = max(sub_ps)
        if max_sub_p < TYPE_LONG_NOISE_GATE:
            flag = True
            (u, ut, _), (v, vt, _) = one_cut[0], one_cut[-1]
            if ii > 0:
                prev_u, prev_ut = cuts[ii - 1][-1][:2]
                if (
                    router.MAP_routing(prev_u, u, prev_ut, ut)
                    > TYPE_LONG_NOISE_KEEP_GATE
                ):
                    flag = False
            if flag and ii < len(cuts) - 1:
                succ_v, succ_vt = cuts[ii + 1][0][:2]
                if (
                    router.MAP_routing(v, succ_v, vt, succ_vt)
                    > TYPE_LONG_NOISE_KEEP_GATE
                ):
                    flag = False
            if flag:
                noises += [(x[-1], TYPE_LONG_NOISE) for x in one_cut]
        elif max_sub_p > BW_LIST_GATE:
            black_list = set()
            white_list = set()
            for i in range(len_cut):
                p = max(p for p, idx in zip(sub_ps_raw, sub_idxs) if i in idx)
                if p < TYPE_BLACK_LIST_GATE:
                    black_list.add(i)
                    noises.append(
                        (one_cut[i][-1], TYPE_BLACK_LIST_NOISE)
                    )
                else:
                    ps = [
                        p
                        for p, idx in zip(sub_ps_raw, sub_idxs)
                        if i in idx and p >= 0.01
                    ]
                    if (
                        len(ps) >= min(len_cut * WHITE_LIST_LENGTH_GATE, 2)
                        and np.mean(ps) > WHITE_LIST_MEAN_GATE
                        and max(ps) > WHITE_LIST_MAX_GATE
                    ):
                        white_list.add(i)

            opt_sub_p, opt_sub_idx = max(
                (x for x in zip(sub_ps, sub_idxs) if x[0] > OPT_SUB_COEFF * max_sub_p),
                key=lambda x: (len(x[1]), x[0]),
            )

            for i in set(range(len_cut)) - set(opt_sub_idx) - black_list - white_list:
                noises.append(
                    (one_cut[i][-1], TYPE_OUT_OF_SUBSET_NOISE)
                )
            if DO_RECALL_ATTEMPT:
                for i, j in zip(opt_sub_idx, opt_sub_idx[1:]):
                    u, ut, _ = one_cut[i]
                    v, vt, _ = one_cut[j]
                    inter_nodes, _ = router.MAP_routing(u, v, ut, vt, return_route=True)
                    inter_camera_nodes = [
                        node for node in inter_nodes if "camera" in G.nodes[node]
                    ]
                    if inter_camera_nodes:
                        recall_attempts.append(
                            ((u, ut), (v, vt), inter_camera_nodes)
                        )
    return noises, recall_attempts


def noise_detect_unit(rids):
    """
    噪声检测的入口函数
    """
    points = [(records[i]["node_id"], records[i]["time"], i) for i in rids]
    # 1. 合并同一node多个邻近tm的情况
    points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
    points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
    # 2. 将points切分为多段, 视为多段轨迹
    points.sort(key=lambda x: x[1])
    cuts = cut_distant_points(points, tm_gap_gate=TM_GAP_GATE)
    return detect_many_noise(cuts)


def noise_detect_unit_batch(rids):
    return [noise_detect_unit(i) for i in rids]


def calculate_ave_f(idxs):
    car1 = [f_car[i] for i in idxs]
    car1 = mean(car1, True)
    plate1 = [x for x in (f_plate[i] for i in idxs) if x is not None]
    if plate1:
        plate1 = mean(plate1, True)
    else:
        plate1 = None
    return car1, plate1


def sim_filter(car1, plate1, candidates, sim_gate=0.7):
    # 过滤出与给定车身特征和车牌特征相似度较高的噪声点
    candidates_filter = []
    for noise in candidates:
        idxs2 = noise[1]
        car2 = [f_car[i] for i in idxs2]
        plate2 = [f_plate[i] for i in idxs2]
        car2 = mean(car2, True)
        plate2 = [x for x in plate2 if x is not None]
        if plate2:
            plate2 = mean(plate2, True)
        else:
            plate2 = None
        sim_car = car1 @ car2
        if plate1 is not None and plate2 is not None:
            sim_plate = plate1 @ plate2
            sim = 0.2 * sim_car + 0.8 * sim_plate
        else:
            sim = sim_car
        if sim > sim_gate:
            candidates_filter.append(noise)
    return candidates_filter


def recall_unit(recall_attempts, cr_idxs, node_to_noises):
    """点缺失召回"""
    car1, plate1 = None, None
    accept_recalls = []
    for tmp in recall_attempts:
        if len(tmp) == 3:
            (u, ut), (v, vt), inter_camera_nodes = tmp
            p_base = None
            for node in inter_camera_nodes:
                candidates = [
                    noise for noise in node_to_noises[node] if ut < noise[0] < vt
                ]
                if not candidates:
                    continue
                if car1 is None:
                    car1, plate1 = calculate_ave_f(cr_idxs)
                candidates_filter = sim_filter(car1, plate1, candidates, sim_gate=0.7)
                if candidates_filter:
                    if p_base is None:
                        p_base = router.MAP_routing(u, v, ut, vt)
                    for tm, idxs in candidates_filter:
                        p_new = sqrt(
                            router.MAP_routing(u, node, ut, tm)
                            * router.MAP_routing(node, v, tm, vt)
                        )
                        t = (
                            p_new * (1 - MISS_SHOT_P) - p_base * MISS_SHOT_P
                        )  # 若不召回该点, 概率为初始两点间的路径概率*漏拍率; 若召回, 概率为三点路径概率*(1-漏拍率)
                        if t > 0:
                            accept_recalls.append(
                                (idxs, t)
                            )  # 记录该recall带来的收益，因为可能出现同一个Noise被多个recall采用的情况，这时需要决定这noise给谁用
        else:  # 车在路口被拍到多次的情形, 尝试找噪声点与现有点合体
            node, tm = tmp
            candidates = [
                noise
                for noise in node_to_noises[node]
                if abs(tm - noise[0]) < ADJ_RANGE / 2
            ]
            if not candidates:
                continue
            if car1 is None:
                car1, plate1 = calculate_ave_f(cr_idxs)
            candidates_filter = sim_filter(car1, plate1, candidates, sim_gate=0.78)
            for tm, idxs in candidates_filter:
                accept_recalls.append((idxs, 0))  # 合并到同一路口的多个点并不影响轨迹概率, 收益为0
    return accept_recalls


def recall_unit_batch(cid_and_recall_attempts, cid_to_rids, node_to_noises):
    return [
        recall_unit(recall_attempts, cid_to_rids[cid], node_to_noises)
        for cid, recall_attempts in cid_and_recall_attempts
    ]


def update_f_emb(labels, vid_to_cid):
    """
    噪声检测+点缺失召回的主函数
    """
    global f_emb

    cid_to_rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

    # 噪声检测
    logging.info("detecting noise...")
    results = tqdm_process_map(
        noise_detect_unit,
        cid_to_rids.values(),
        max_workers=NUM_WORKERS,
        chunksize=400,
        dynamic_ncols=True,
    )
    # 检测出的噪声
    cid_to_noises = {}
    # 抛出的召回请求
    cid_to_recall_attempts = {}
    for cid, result in zip(cid_to_rids.keys(), results):
        noises, recall_attempts = result
        if noises:
            cid_to_noises[
                cid
            ] = noises  # cid_to_noises: noises is list of (idxs, noise_type) (idxs是一个list, 为合并的点)
        if recall_attempts:
            cid_to_recall_attempts[cid] = recall_attempts

    # 点缺失召回
    cid_to_accept_recalls = defaultdict(list)
    if DO_RECALL_ATTEMPT:
        logging.info("recalling...")
        node_to_noises = defaultdict(list)  # 记录每个node下的noise
        for idxs in (noise[0] for noises in cid_to_noises.values() for noise in noises):
            tms = [records[i]["time"] for i in idxs]
            node = records[idxs[0]]["node_id"]
            node_to_noises[node].append((np.mean(tms), idxs))
        for rids in cid_to_rids.values():  # 把单点的类也作为点缺失召回的来源
            if len(rids) == 1:
                r = records[idxs[0]]
                node_to_noises[r["node_id"]].append((r["time"], rids))
        batch_size = 500
        ls = list(cid_to_recall_attempts.items())
        results = sum(
            process_map(
                recall_unit_batch,
                [
                    (ls[i : i + batch_size], cid_to_rids, node_to_noises)
                    for i in range(0, len(ls), batch_size)
                ],
                num_workers=NUM_WORKERS,
                unpack=True,
            ),
            [],
        )
        # 可能有多个cluster同时接受同一个idxs作为recall，记录每个idxs放到各个cluster中的收益，将idxs放到收益最大的cluster中
        idxs_to_cid_reward = defaultdict(list)
        for cid, accept_recalls in zip(cid_to_recall_attempts.keys(), results):
            if accept_recalls:
                for idxs, reward in accept_recalls:
                    idxs_to_cid_reward[tuple(idxs)].append((cid, reward))
        for idxs, cid_reward in idxs_to_cid_reward.items():
            cid_to_accept_recalls[
                max(cid_reward, key=lambda x: x[1])[0]
            ] += idxs  # cid_to_accept_recalls: accept_recalls is [idx] (idx已经展开)

    recalled_noises = []  # 被recall的noise不再视为noise做外推，记下来
    to_update = []
    for cid, idxs in cid_to_accept_recalls.items():
        recalled_noises += idxs
        # tmp = [f_emb[i] for i in cid_to_rids[cid]]  # 算均值时噪声不参与计算
        rids = cid_to_rids[cid]
        nids = [y for x in cid_to_noises.get(cid, []) for y in x[0]]
        t = set(rids) - set(nids)
        if len(t) >= len(rids) / 2:
            tmp = mean([f_emb[i] for i in t])
        else:
            tmp = mean([f_emb[i] for i in rids])
        to_update.append((tmp, idxs))

    for tmp, idxs in to_update:  # 被recall的noise，embedding修改为要去的类的平均embedding
        for i in idxs:
            f_emb[i] = length_clamp(tmp)
    
    # 去噪
    strong_noise_types = {
        TYPE_ONE_NOISE,
        TYPE_TWO_NOISE,
        TYPE_LONG_NOISE,
        TYPE_BLACK_LIST_NOISE,
    }
    ordinary_noise_types = {TYPE_OUT_OF_SUBSET_NOISE}  # SINGLE ClUSTER类型的noise不做处理
    one_iter_monitor = []  # 记录所有噪声被推出去前, 离类中心的距离
    one_iter_vc_monitor = []  # 记录vid_to_cid中的cid的类里, 噪声被推出去前, 离类中心的距离
    vcids = set(vid_to_cid.values())
    for cid, noises in cid_to_noises.items():  # 剩下的noise往外推
        is_vc = cid in vcids
        strong_noises = [
            x[0]
            for x in noises
            if x[1] in strong_noise_types and x[0] not in recalled_noises
        ]
        ordinary_noises = [
            x[0]
            for x in noises
            if x[1] in ordinary_noise_types and x[0] not in recalled_noises
        ]
        strong_noises = set(sum(strong_noises, []))
        ordinary_noises = set(sum(ordinary_noises, []))
        noises = strong_noises | ordinary_noises
        pos = cid_to_rids[cid]
        tmp = [f_emb[i] for i in pos if i not in noises]  # 远离非噪声点
        if tmp:
            tmp = mean(tmp)
        else:
            tmp = mean([f_emb[i] for i in pos])  # 一个类全部被识别为噪声点，互相远离
        for noises, push in [
            (strong_noises, STRONG_NOISE_PUSH),
            (ordinary_noises, ORDINARY_NOISE_PUSH),
        ]:
            for i in noises:
                f_delta = f_emb[i] - tmp
                length = np.linalg.norm(f_delta)
                f_emb[i] = length_clamp(
                    f_emb[i]
                    + push * f_delta / (length + 1e-12) * max(length, MIN_PUSH_LENGTH)
                )
                assert np.all(np.isfinite(f_emb[i]))
                # one_iter_monitor.append(length)  # 记录推之前离类中心的距离
                # if is_vc:
                #     one_iter_vc_monitor.append(length)
    return (
        cid_to_noises,
        cid_to_accept_recalls,
        cid_to_recall_attempts,
        one_iter_monitor,
        one_iter_vc_monitor,
    )


def evaluate_noise_detect_and_recall(
    labels, vid_to_cid, cid_to_noises, cid_to_accept_recalls, cid_to_recall_attempts
):
    """评估噪声检测和点缺失召回的效果"""
    def noise_type_name(noise_type):
        if noise_type == TYPE_ORDINARY_NOISE:
            return "ordinary"
        elif noise_type == TYPE_STRONG_NOISE:
            return "strong"
        elif noise_type == TYPE_ONE_NOISE:
            return "one_point"
        elif noise_type == TYPE_TWO_NOISE:
            return "two_point"
        elif noise_type == TYPE_LONG_NOISE:
            return "many_point"
        elif noise_type == TYPE_BLACK_LIST_NOISE:
            return "black_list"
        elif noise_type == TYPE_OUT_OF_SUBSET_NOISE:
            return "out_of_subset"
        elif noise_type == TYPE_SINGLE_CLUSTER:
            return "single_cluster"

    logging.info("------------------")
    noises_for_each_type = defaultdict(list)
    noise_list = [noise for noises in cid_to_noises.values() for noise in noises]
    print("total detected noise: ", len(noise_list))
    print("total accepted recall:", sum(map(len, cid_to_accept_recalls.values())))
    for noise, noise_type in noise_list:
        noises_for_each_type[noise_type].extend(noise)
    noise_type_dict = {
        idx: noise_type
        for noise_type, noises in noises_for_each_type.items()
        for idx in noises
    }

    vcids = list(vid_to_cid.values())
    noises_for_each_type_vc = defaultdict(list)
    noise_list_vc = [
        noise
        for cid, noises in cid_to_noises.items()
        if cid in vcids
        for noise in noises
    ]
    print("detected noise in vehicle clusters: ", len(noise_list_vc))
    print(
        "accepted recall in vehicle clusters:",
        sum(
            map(
                len,
                (idxs for cid, idxs in cid_to_accept_recalls.items() if cid in vcids),
            )
        ),
    )
    for noise, noise_type in noise_list_vc:
        noises_for_each_type_vc[noise_type].extend(noise)
    noise_types = [
        TYPE_SINGLE_CLUSTER,
        TYPE_ONE_NOISE,
        TYPE_TWO_NOISE,
        TYPE_LONG_NOISE,
        TYPE_BLACK_LIST_NOISE,
        TYPE_OUT_OF_SUBSET_NOISE,
    ]
    print(
        "                 single_clst one_point   two_point   many_point  black_point out_of_subset"
    )
    print(
        "all clusters     "
        + "".join(f"{len(noises_for_each_type.get(nt, [])):<12d}" for nt in noise_types)
    )
    print(
        "vehicle clusters "
        + "".join(
            f"{len(noises_for_each_type_vc.get(nt, [])):<12d}" for nt in noise_types
        )
    )
    print("\n------------------")

    cid_to_rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

    total_gt = 0
    orig_recall = 0
    orig_miss = 0
    orig_noise = 0

    miss_in_noise = 0
    miss_in_attempt = 0
    miss_in_noise_attempt = 0

    total_detected_noise = sum(len(x[0]) for x in cid_to_noises.values())
    total_accepted_recall = sum(len(x) for x in cid_to_accept_recalls.values())
    detected_true_noise = 0
    undetected_true_noise = 0
    fake_noise = 0
    accepted_true_recall = 0
    undo_true_recall = 0
    fake_recall = 0

    noise_type_to_performance_cnt = {
        noise_type: {
            "true_noise": 0,
            "fake_noise": 0,
            "true_recall": 0,
            "fake_recall": 0,
        }
        for noise_type in noises_for_each_type
    }
    noise_idxs = {idx for noise_type in noise_list for idx in noise_type[0]}
    for vid, cid in vid_to_cid.items():
        vrids = set(vid_to_rids[vid])
        crids = set(cid_to_rids[cid])
        recall_idxs = set(cid_to_accept_recalls.get(cid, []))
        # 原召回
        recall_set = vrids & crids
        # 原缺失
        miss_set = vrids - crids
        # 原噪声
        noise_set = crids - vrids
        # 真噪声：原噪声被分为噪声的
        detected_noise_set = noise_set & noise_idxs
        # 假噪声：原召回被分为噪声的
        fake_noise_set = recall_set & noise_idxs
        # 真召回：原缺失被召回的
        accepted_true_recall_set = miss_set & recall_idxs
        # 假召回：错误召回
        fake_recall_set = recall_idxs - miss_set

        for idx in detected_noise_set:
            noise_type = noise_type_dict[idx]
            noise_type_to_performance_cnt[noise_type]["true_noise"] += 1
        for idx in fake_noise_set:
            noise_type = noise_type_dict[idx]
            noise_type_to_performance_cnt[noise_type]["fake_noise"] += 1
        for idx in accepted_true_recall_set:
            noise_type = noise_type_dict[idx]
            noise_type_to_performance_cnt[noise_type]["true_recall"] += 1
        for idx in fake_recall_set:
            noise_type = noise_type_dict[idx]
            noise_type_to_performance_cnt[noise_type]["fake_recall"] += 1

        total_gt += len(vrids)
        orig_recall += len(recall_set)
        orig_miss += len(miss_set)
        orig_noise += len(noise_set)

        miss_in_noise += len(miss_set & noise_idxs)
        attempts = cid_to_recall_attempts.get(cid, [])
        if attempts:
            node_to_miss = defaultdict(list)
            for i in miss_set:
                node_to_miss[records[i]["node_id"]].append((records[i]["time"], i))
            for tmp in attempts:
                if len(tmp) == 2:
                    node, attempt_tm = tmp
                    tms = [
                        (tm, i)
                        for tm, i in node_to_miss[node]
                        if abs(attempt_tm - tm) < ADJ_RANGE / 2
                    ]
                    if tms:
                        miss_in_attempt += len(tms)
                        _, idxs = list(zip(*tms))
                        miss_in_noise_attempt += len(set(idxs) & noise_idxs)
                else:
                    (u, ut), (v, vt), inter_nodes = tmp
                    for node in inter_nodes:
                        tms = [(tm, i) for tm, i in node_to_miss[node] if ut < tm < vt]
                        if tms:
                            miss_in_attempt += len(tms)
                            _, idxs = list(zip(*tms))
                            miss_in_noise_attempt += len(set(idxs) & noise_idxs)

        detected_true_noise += len(detected_noise_set)
        undetected_true_noise += len(noise_set) - len(detected_noise_set)
        fake_noise += len(fake_noise_set)

        accepted_true_recall += len(accepted_true_recall_set)
        undo_true_recall += len(miss_set) - len(accepted_true_recall_set)
        fake_recall += len(fake_recall_set)

    width = 25
    # print(ljust("真值总数", width), total_gt)
    # print(ljust("原召回", width), orig_recall)
    # print(ljust("原缺失", width), orig_miss)
    # print(
    #     ljust(" - 被检测为它类噪声", width),
    #     miss_in_noise,
    #     f"{miss_in_noise/orig_miss*100:.2f}%",
    # )
    # print(
    #     ljust(" - 位于途径摄像头", width),
    #     miss_in_attempt,
    #     f"{miss_in_attempt/orig_miss*100:.2f}%",
    # )
    # print(
    #     ljust(" - 上述二者交集", width),
    #     miss_in_noise_attempt,
    #     f"{miss_in_noise_attempt/orig_miss*100:.2f}%",
    # )
    # print(ljust("原噪声", width), orig_noise)
    # print("噪声检测")
    # print(ljust(" - 总数：", width), total_detected_noise)
    # print(
    #     ljust(" - 正确：", width),
    #     detected_true_noise,
    #     f"P={detected_true_noise / (detected_true_noise + fake_noise + 1e-6)*100:.2f}% "
    #     f"R={detected_true_noise / orig_noise*100:.2f}%",
    # )
    # print(ljust(" - 错误：", width), fake_noise)
    # print(ljust(" - 未检测到：", width), undetected_true_noise)
    # print("召回")
    # print(ljust(" - 总数：", width), total_accepted_recall)
    # print(
    #     ljust(" - 正确：", width),
    #     accepted_true_recall,
    #     f"P={ accepted_true_recall / (accepted_true_recall + fake_recall + 1e-6)*100:.2f}% "
    #     f"R={accepted_true_recall / orig_miss*100:.2f}%",
    # )
    # print(ljust(" - 错误：", width), fake_recall)
    # print(ljust(" - 未召回：", width), undo_true_recall)
    print(" " * 15 + "true_noise  fake_noise  true_recall fake_recall")
    for noise_type, performance_cnt in noise_type_to_performance_cnt.items():
        noise_type = noise_type_name(noise_type)
        print(
            noise_type.ljust(15)
            + "".join(f"{a:<12d}" for a in performance_cnt.values())
        )


def ave_f_unit(rids):
    """计算平均特征"""
    if len(rids) == 1:
        return f_car[i], f_plate[i]
    else:
        fs_car = [f_car[i] for i in rids]
        ave_car = sum(fs_car) / len(fs_car)
        fs_plate = [x for x in (f_plate[i] for i in rids) if x is not None]
        if fs_plate:
            ave_plate = sum(fs_plate) / len(fs_plate)
        else:
            ave_plate = None
        return ave_car, ave_plate


def tms_adj_range(tms, adj_range):
    """
    对于一个单点tm, 返回(tm-adj_range, tm+adj_range)作为其邻近时间范围
    对于一个list tms, 返回每个单点tm邻近时间范围的并集
    要求tms已排好序
    """
    tm = tms[0]
    adj_ranges = [[max(tm - adj_range, 0), tm + adj_range]]
    for tm in tms[1:]:
        tm_m = tm - adj_range
        tm_p = tm + adj_range
        if tm_m <= adj_ranges[-1][1]:
            adj_ranges[-1][1] = tm_p
        else:
            adj_ranges.append([tm_m, tm_p])  # minus, plus
    return adj_ranges


def merge_cluster_unit(c, ncs, cid_to_rids):
    """块缺失召回实现"""
    # 按余弦相似度过滤ncs
    idxs1 = cid_to_rids[c]
    car1 = [f_car[i] for i in idxs1]
    plate1 = [f_plate[i] for i in idxs1]
    car1 = mean(car1, True)
    plate1 = [x for x in plate1 if x is not None]
    if plate1:
        plate1 = mean(plate1, True)
    else:
        plate1 = None

    nidxs_filter = []
    for nc in ncs:
        idxs2 = cid_to_rids[nc]
        for i in idxs2:
            car2 = f_car[i]
            plate2 = f_plate[i]
            car2 /= np.linalg.norm(car2) + 1e-12
            if plate2 is not None:
                plate2 /= np.linalg.norm(plate2) + 1e-12
            sim_car = car1 @ car2
            if plate1 is not None and plate2 is not None:
                sim_plate = plate1 @ plate2
                sim = 0.2 * sim_car + 0.8 * sim_plate
            else:
                sim = sim_car
            if sim > MERGE_CLUSTER_SIM_GATE:
                nidxs_filter.append(i)
    if not nidxs_filter:
        return []

    # 从时空角度考虑是否合并
    points = [(records[i]["node_id"], records[i]["time"], i) for i in cid_to_rids[c]]
    tm_ranges = tms_adj_range(
        sorted([x[1] for x in points]), adj_range=MERGE_CLUSTER_ADJ_RANGE
    )
    points_nc = [(records[i]["node_id"], records[i]["time"], i) for i in nidxs_filter]
    points_nc_filter = []  # 选出时间落在大类中至少某一点附近的点
    for p in points_nc:
        t = p[1]
        flag = False
        for min_t, max_t in tm_ranges:
            if min_t < t < max_t:
                flag = True
                break
        if flag:
            points_nc_filter.append(p)
    if not points_nc_filter:
        return []
    points_nc = points_nc_filter

    points_all = points + points_nc  # 将这些点合并后做噪声检测(参数设置得比正常噪声检测更严格)，未被检测为噪声的允许合并
    points_all = merge_tm_adj_points(points_all, adj_range=ADJ_RANGE)
    points_all = merge_tm_adj_points(points_all, adj_range=ADJ_RANGE)
    points_all.sort(key=lambda x: x[1])
    cuts = cut_distant_points(points_all, tm_gap_gate=TM_GAP_GATE)

    noises = []
    long_cuts = []
    for i, one_cut in enumerate(cuts):
        if len(one_cut) == 1:
            noises.append(one_cut[0][-1])
        elif len(one_cut) == 2:
            (u, ut, _), (v, vt, _) = one_cut
            p = router.MAP_routing(u, v, ut, vt)
            if p < TYPE_TWO_NOISE_GATE_2:
                noises += [x[-1] for x in one_cut]
        else:
            long_cuts.append(one_cut)
    for one_cut in long_cuts:
        len_cut = len(one_cut)
        p_dict = {}
        sub_ps_raw = []
        sub_ps = []
        sub_idxs = []
        for sub_idx in subsets(
            list(range(len_cut)),
            k=ceil(len_cut * SUBSET_SIZE_RATIO_2),
        ):
            ps = []
            for i, j in zip(sub_idx, sub_idx[1:]):
                p = p_dict.get((i, j), None)
                if p is None:
                    u, ut = one_cut[i][:2]
                    v, vt = one_cut[j][:2]
                    p = router.MAP_routing(u, v, ut, vt)
                    p_dict[i, j] = p
                ps.append(p)
            p = np.exp(np.mean(np.log(ps)))
            sub_ps_raw.append(p)
            sub_ps.append(p * len(sub_idx) / len_cut)
            sub_idxs.append(sub_idx)

        max_sub_p = max(sub_ps)
        if max_sub_p < TYPE_LONG_NOISE_GATE_2:
            noises += [x[-1] for x in one_cut]
        else:
            black_list = set()
            white_list = set()
            for i in range(len_cut):
                p = max(p for p, idx in zip(sub_ps_raw, sub_idxs) if i in idx)
                if p < TYPE_BLACK_LIST_GATE_2:
                    black_list.add(i)
                    noises.append(one_cut[i][-1])
                else:
                    ps = [
                        p
                        for p, idx in zip(sub_ps_raw, sub_idxs)
                        if i in idx and p >= 0.01
                    ]
                    if (
                        len(ps) >= min(len_cut * WHITE_LIST_LENGTH_GATE_2, 3)
                        and np.mean(ps) > WHITE_LIST_MEAN_GATE_2
                        and max(ps) > WHITE_LIST_MAX_GATE_2
                    ):
                        white_list.add(i)
            opt_sub_p, opt_sub_idx = max(
                (x for x in zip(sub_ps, sub_idxs) if x[0] > OPT_SUB_COEFF * max_sub_p),
                key=lambda x: (len(x[1]), x[0]),
            )
            for i in set(range(len_cut)) - set(opt_sub_idx) - black_list - white_list:
                noises.append(one_cut[i][-1])
    noise_idxs = {idx for idxs in noises for idx in idxs}
    orig_idxs = {x[-1] for x in points}
    merge_point_idxs = set()
    for p in points_all:
        idxs = {x for x in p[-1]}
        tmp = idxs - orig_idxs
        if len(tmp) < len(
            idxs
        ):
            for idx in tmp:
                merge_point_idxs.add(idx)
    accept_idxs = {x[-1] for x in points_nc} - noise_idxs - merge_point_idxs
    return list(accept_idxs)


def merge_cluster_batch(batch, cid_to_rids):
    return [merge_cluster_unit(c, ncs, cid_to_rids) for c, ncs in batch]


def merge_clusters(labels, vid_to_cid, gpus):
    """块缺失召回主函数"""
    logging.info("cluster merging...")
    start_time = time.time()

    cid_to_rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

    one_iter_monitor = []  # 记录所有缺失被召回后, 离类中心的距离
    one_iter_vc_monitor = []  # 记录vid_to_cid中的cid的类里, 缺失被召回后, 离类中心的距离
    vcids = set(vid_to_cid.values())
    tf_cnt = []
    for nn in range(3):
        cs_big = []
        cs_small = []
        if nn == 0:
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 10 < t <= 20:
                    cs_big.append(c)
                elif t <= 10:
                    cs_small.append(c)
        elif nn == 1:
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 20 < t: # <= 30:
                    cs_big.append(c)
                elif t <= 20:
                    cs_small.append(c)
        elif nn == 2:
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 30 < t:  # <= 40:
                    cs_big.append(c)
                elif t <= 30:
                    cs_small.append(c)

        # 计算平均特征
        logging.info("averaging...")
        results = process_map(
            ave_f_unit,
            [cid_to_rids[c] for c in cs_big],
            num_workers=NUM_WORKERS,
            disable=True,
        )
        car_query = np.asarray([x[0] for x in results])
        tmp = [(x[1], c) for x, c in zip(results, cs_big) if x[1] is not None]
        plate_query = np.asarray([x[0] for x in tmp])
        plate_query_c = [x[1] for x in tmp]

        logging.info("averaging...")
        results = process_map(
            ave_f_unit,
            [cid_to_rids[c] for c in cs_small],
            num_workers=NUM_WORKERS,
            disable=True,
        )
        car_gallery = np.asarray([x[0] for x in results])
        tmp = [(x[1], c) for x, c in zip(results, cs_small) if x[1] is not None]
        plate_gallery = np.asarray([x[0] for x in tmp])
        plate_gallery_c = [x[1] for x in tmp]

        # 按f_car和f_plate分别搜索topk
        c_to_nc = defaultdict(set)
        car_topk_idxs = (
            FlatSearcher(feat_len=F_DIM, gpus=gpus)
            .search_by_topk(query=car_query, gallery=car_gallery, topk=MERGE_CAR_TOPK)[
                1
            ]
            .tolist()
        )  # query * topk
        for c, idxs in zip(cs_big, car_topk_idxs):
            for i in idxs:
                c_to_nc[c].add(cs_small[i])
        if len(plate_query) and len(plate_gallery):
            plate_topk_idxs = (
                FlatSearcher(feat_len=F_DIM, gpus=gpus)
                .search_by_topk(
                    query=plate_query, gallery=plate_gallery, topk=MERGE_PLATE_TOPK
                )[1]
                .tolist()
            )
            for c, idxs in zip(plate_query_c, plate_topk_idxs):
                for i in idxs:
                    c_to_nc[c].add(plate_gallery_c[i])

        # 按余弦相似度过滤neighboring_clusters，然后从时空角度考虑是否合并
        logging.info("merging...")
        tmp = list(c_to_nc.items())
        batch_size = 100
        results = sum(
            process_map(
                merge_cluster_batch,
                [
                    [tmp[i : i + batch_size], cid_to_rids]
                    for i in range(0, len(tmp), batch_size)
                ],
                num_workers=NUM_WORKERS,
                unpack=True,
                disable=True
            ),
            [],
        )
        accept_idx_to_cs = defaultdict(list)
        for c, accept_idxs in zip(c_to_nc.keys(), results):
            for idx in accept_idxs:
                accept_idx_to_cs[idx].append(c)
        c_to_accept_idxs = defaultdict(list)
        for idx, cs in accept_idx_to_cs.items():
            if len(cs) == 1:
                c_to_accept_idxs[cs[0]].append(idx)
            else:
                c_to_accept_idxs[random.sample(cs, 1)[0]].append(idx)

        # 统计合并的正确数和错误数
        p_old, r_old, f_old, e_old, vid_to_cid = evaluate(records, labels, log=False)
        true_cnt = 0
        fake_cnt = 0
        for v, c in vid_to_cid.items():
            true_idxs = vid_to_rids[v]
            for idx in c_to_accept_idxs.get(c, []):
                if idx in true_idxs:
                    true_cnt += 1
                else:
                    fake_cnt += 1
        logging.info(f"true merge cnt: {true_cnt}, fake merge cnt: {fake_cnt}")
        tf_cnt.append([true_cnt, fake_cnt])

        # 更新f_emb
        for c, idxs in c_to_accept_idxs.items():
            for idx in idxs:
                labels[idx] = c
        for c, idxs in c_to_accept_idxs.items():
            c_embs = np.array([f_emb[i] for i in cid_to_rids[c]])
            tmp = np.mean(c_embs, axis=0)
            r = np.linalg.norm(c_embs - tmp, axis=0).mean()  # 该类的平均半径
            assert r > 0
            # 将f_emb朝要合并的类的中心吸, 使得至中心的距离等于该类的平均半径, 若已经小于此距离则不动
            for i in idxs:
                f_delta = f_emb[i] - tmp
                length = np.linalg.norm(f_delta)
                if length > r:
                    f_emb[i] = length_clamp(tmp + f_delta * r / length)
            # 记录吸完之后离类中心的距离
            diss = [np.linalg.norm(f_emb[i] - tmp) for i in idxs]
            one_iter_monitor += diss
            if c in vcids:
                one_iter_vc_monitor += diss

        p, r, f, e, _ = evaluate(records, labels, log=False)
        logging.info(
            f"precision: {(p-p_old):+.6f}\nrecall:    {(r-r_old):+.6f}\nfscore:    {(f-f_old):+.6f}\nexpansion: {(e-e_old):+.6f}"
        )
    logging.info(f"merging consume time: {time.time() - start_time:.2f}")
    
    table = Table(4, 3)
    table[0, 1] = "true_recall"
    table[0, 2] = "fake_recall"
    for i, (j, k) in enumerate(tf_cnt):
        table[i + 1, 0] = f"round_{i}"
        table[i + 1, 1] = j
        table[i + 1, 2] = k
    print(table)
    return labels, one_iter_monitor, one_iter_vc_monitor


if __name__ == "__main__":

    #  ========  读配置  ======== #

    tgttop, target, cfg = read_config()
    G_path = f"../data_interface/G_{tgttop}.pkl"
    F_DIM = cfg['pca']
    record_path = f"../data_interface/records_pca_{F_DIM}_{tgttop}_{target}.pkl"
    camera_path = f"../data_interface/r2cameras_{tgttop}.pkl"
    main_cfg_path = cfg["regions"][target]["main_config"]
    cfg = yaml.load(open(main_cfg_path, "r", encoding="utf8"), Loader=yaml.SafeLoader)

    random.seed(cfg["seed"])
    exp_name = cfg["name"].replace("%t", time.strftime("%y%m%d_%H%M%S"))

    setproctitle(exp_name + "@yfd")
    ref_name = cfg.get("ref_name", "")
    debug_phase = cfg["debug_phase"]
    assert debug_phase in [0, 1, 2]
    start_from_last = cfg["start_from_last"]
    DO_RECALL_ATTEMPT = cfg["do_recall_attempt"]
    MERGE_CLUSTER_SIM_GATE = cfg["merge_cluster_sim_gate"]
    MISS_SHOT_P = cfg["miss_shot_p"]
    ADJ_RANGE = cfg["adj_range"]
    TM_GAP_GATE = cfg["tm_gap_gate"]
    MERGE_CLUSTER_ADJ_RANGE = cfg["merge_cluster_adj_range"]
    STRONG_NOISE_PUSH = cfg["strong_noise_push"]
    ORDINARY_NOISE_PUSH = cfg["ordinary_noise_push"]
    MIN_PUSH_LENGTH = cfg["min_push_length"]
    MERGE_CAR_TOPK = cfg["merge_car_topk"]
    MERGE_PLATE_TOPK = cfg["merge_plate_topk"]
    TYPE_ONE_NOISE_KEEP_TOTAL_LEN = cfg["type_one_noise_keep_total_len"]
    TYPE_ONE_NOISE_KEEP_GATE = cfg["type_one_noise_keep_gate"]
    SUBSET_SIZE_RATIO_2 = cfg["subset_size_ratio_2"]
    TYPE_TWO_NOISE_GATE = cfg["type_two_noise_gate"]
    TYPE_TWO_NOISE_GATE_2 = cfg["type_two_noise_gate_2"]
    TYPE_TWO_NOISE_KEEP_GATE = cfg["type_two_noise_keep_gate"]
    TYPE_LONG_NOISE_GATE = cfg["type_long_noise_gate"]
    TYPE_LONG_NOISE_GATE_2 = cfg["type_long_noise_gate_2"]
    TYPE_LONG_NOISE_KEEP_GATE = cfg["type_long_noise_keep_gate"]
    BW_LIST_GATE = cfg["bw_list_gate"]
    WHITE_LIST_LENGTH_GATE = cfg["white_list_length_gate"]
    WHITE_LIST_LENGTH_GATE_2 = cfg["white_list_length_gate_2"]
    WHITE_LIST_MEAN_GATE = cfg["white_list_mean_gate"]
    WHITE_LIST_MEAN_GATE_2 = cfg["white_list_mean_gate_2"]
    WHITE_LIST_MAX_GATE = cfg["white_list_max_gate"]
    WHITE_LIST_MAX_GATE_2 = cfg["white_list_max_gate_2"]
    TYPE_BLACK_LIST_GATE = cfg["type_black_list_gate"]
    TYPE_BLACK_LIST_GATE_2 = cfg["type_black_list_gate_2"]
    OPT_SUB_COEFF = cfg["opt_sub_coeff"]
    OPT_SUB_COEFF_2 = cfg["opt_sub_coeff_2"]
    N_iter = cfg["num_iter"]
    GPUS = tuple(int(i) for i in (str(cfg["cuda"])).split(","))
    NUM_WORKERS = cfg["num_workers"]
    MAX_LENGTH = cfg.get("max_length", 1e3)

    #  ========  开log目录  ======== #

    if os.path.exists(f"log/{exp_name}"):
        if input(f"Folder [{exp_name}] already exists, continue?(y|N)") != "y":
            exit()
    os.makedirs(f"log/{exp_name}/labels", exist_ok=True)
    os.makedirs(f"log/{exp_name}/f_emb", exist_ok=True)
    if (
        start_from_last
        and ref_name
        and ref_name != exp_name
        and not os.path.exists(f"log/{exp_name}/labels/iter_0.pkl")
    ):
        shutil.copy(f"log/{ref_name}/labels/iter_0.pkl", f"log/{exp_name}/labels")
    
    #  ========  读数据  ======== #

    cameras = pickle.load(open(camera_path, "rb"))[target]  # [{'id', 'node_id', 'xy', 'gps', 'gps_orig'}]
    cameras_dict = {x["id"]: x for x in cameras}
    G = pickle.load(open(G_path, "rb"))

    sample_num = cfg.get("sample_data", -1)
    if sample_num > 0:
        print("sample_num:", sample_num)
        sample_path = f"../data_interface/records_pca_{F_DIM}_{tgttop}_{target}_sample_{sample_num}.pkl"
        if os.path.exists(sample_path):
            records = pickle.load(open(sample_path, "rb"))
        else:
            records = pickle.load(open(record_path, "rb"))
            records = random.sample(
                [r for r in records if r["vehicle_id"] is None], sample_num
                ) + [r for r in records if r["vehicle_id"] is not None]
            pickle.dump(records, open(sample_path, "wb"))
    else:
        records = pickle.load(open(record_path, "rb"))
    for r in records:
        r["node_id"] = cameras_dict[r["camera_id"]]["node_id"]
        del r["camera_id"]

    f_car = [x["car_feature"] for x in records]
    print("f_car:", len(f_car), len(f_car[0]))
    f_plate = [x["plate_feature"] for x in records]
    print("f_plate:", len([x for x in f_plate if x is not None]))
    f_emb = deepcopy(f_car)

    for r in records:
        del r["car_feature"]
        del r["plate_feature"]

    vid_to_rids = defaultdict(list)
    for i, r in enumerate(records):
        t = r["vehicle_id"]
        if t is not None:
            vid_to_rids[t].append(i)

    print("record num:", len(records))
    print("feature dim:", F_DIM)
    print("record num with ground truth:", len([r for r in records if r["vehicle_id"] is not None]))
    print("ground truth vehicle num:", len(set(r["vehicle_id"] for r in records if r["vehicle_id"] is not None)))
    records = {i: r for i, r in enumerate(records)}

    router = Router()

    #  ========  调参  ======== #

    if debug_phase < 2:
        labels = pickle.load(open(f"log/{exp_name}/labels/iter_0.pkl", "rb"))
        precision, recall, fscore, expansion, vid_to_cid = evaluate(records, labels)
        if debug_phase == 0:
            cid_to_noises, cid_to_accept_recalls, cid_to_recall_attempts = update_f_emb(
                labels, vid_to_cid
            )[:3]
            evaluate_noise_detect_and_recall(
                labels,
                vid_to_cid,
                cid_to_noises,
                cid_to_accept_recalls,
                cid_to_recall_attempts,
            )
        else:
            merge_clusters(labels, vid_to_cid, GPUS)
        exit()

    #  ========  跑主实验  ======== #

    all_st = time.time()

    metrics = []
    recipe = cfg.get("recipe", "merge,denoise").split(",")
    assert len(set(recipe) - {"merge", "denoise"}) == 0, set(recipe)
    recipe = (recipe * N_iter)[:N_iter]
    logging.info("Recipe: " + " ".join(i[0] for i in recipe).upper())
    it = list(zip(range(N_iter), recipe))
    labels = None
    if start_from_last:  # 直接读取缓存的迭代结果
        for i in range(100):
            if not os.path.exists(f"log/{exp_name}/f_emb/iter_{i}.pkl"):
                break
        if i > 0:
            f_emb = pickle.load(open(f"log/{exp_name}/f_emb/iter_{i-1}.pkl", "rb"))
            assert len(f_emb) == len(f_car)
            assert f_emb[0].shape == f_car[0].shape
            metrics = json.load(open(f"log/{exp_name}/metrics.json", "r"))
            assert i + 1 >= len(metrics) >= i, (len(metrics), i)
            metrics = metrics[:i]
            it = it[i:]
            for e in tqdm(f_emb, dynamic_ncols=True):
                assert np.all(np.isfinite(e))
                l = np.linalg.norm(e)
                if l >= MAX_LENGTH:
                    e *= MAX_LENGTH / l
            print(f"Skipped to iter {i}")
    f_topks = None
    for i, operation in it:
        logging.info(f"---------- iter {i} operation {operation} -----------")
        if labels is None:
            if start_from_last and os.path.exists(  # 直接读取缓存的迭代结果
                f"log/{exp_name}/labels/iter_{i}.pkl"
            ):
                logging.info("using last clustering results")
                labels = pickle.load(open(f"log/{exp_name}/labels/iter_{i}.pkl", "rb"))
            else:
                logging.info("clustering...")
                start_time = time.time()
                f_out = []
                labels = SigCluster(feature_dims=[F_DIM, F_DIM, F_DIM], gpus=GPUS).fit(
                    (f_car, f_plate, f_emb),
                    weights=[0.1, 0.8, 0.1],
                    similarity_threshold=cfg["cluster_sim_gate"],
                    topK=cfg["cluster_topk"],
                    zipped_data=False,
                    f_topks_in=f_topks,
                    f_topks_out=f_out,
                )
                f_topks = f_out[0][:2] + [None]
                logging.info(f"clustering consume time: {time.time() - start_time:.2f}")
                pickle.dump(labels, open(f"log/{exp_name}/labels/iter_{i}.pkl", "wb"))  # 缓存每次迭代的labels

        precision, recall, fscore, expansion, vid_to_cid = evaluate(records, labels)
        metrics.append((precision, recall, fscore, expansion, time.time() - all_st))
        json.dump(metrics, open(f"log/{exp_name}/metrics.json", "w"))  # 记录每次迭代的评价指标

        if operation == "merge":
            _, one_iter_monitor, one_iter_vc_monitor = merge_clusters(
                labels=labels, vid_to_cid=vid_to_cid, gpus=GPUS
            )
        elif operation == "denoise":
            (
                cid_to_noises,
                cid_to_accept_recalls,
                cid_to_recall_attempts,
                one_iter_monitor,
                one_iter_vc_monitor,
            ) = update_f_emb(labels=labels, vid_to_cid=vid_to_cid)
            evaluate_noise_detect_and_recall(
                labels,
                vid_to_cid,
                cid_to_noises,
                cid_to_accept_recalls,
                cid_to_recall_attempts,
            )
        pickle.dump(f_emb, open(f"log/{exp_name}/f_emb/iter_{i}.pkl", "wb"))  # 缓存每次迭代的f_emb

        labels = None
