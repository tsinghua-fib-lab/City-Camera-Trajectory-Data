"""
根据高德轨迹路网匹配的结果, 计算道路车速
"""
import os
import pickle
import random
import sys
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from shapely.geometry import Point
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

sys.path.append("..")
from toputils import *

MIN_SPEED = 1
MIN_SPEED2 = 3
MAX_SPEED = 33
MAX_SPEED2 = 28
DIS_GATE = 8
DIS_GATE2 = 8
FAKE_SPEED = 10
ADJ_GATE = 1
METHOD_WEIGHT = [0.45, 0.35, 0.2]

LEVEL2SPEED = {
    "motorway": 14.22,
    "motorway_link": 10.60,
    "primary": 8.57,
    "unclassified": 4.70,
    "trunk": 10.70,
    "secondary": 7.37,
    "residential": 4.69,
    "trunk_link": 10.21,
    "tertiary": 6.26,
    "primary_link": 6.88,
    "living_street": 5.70,
    "secondary_link": 6.98,
    "tertiary_link": 6.72,
}  # 历史经验各等级道路平均速度粗略值


def get_level2speed(eid2level, eid2s, do_print=False):
    level2ss = defaultdict(list)
    for eid, s in eid2s.items():
        level = eid2level[eid]
        if isinstance(level, str):
            level2ss[level].append(s)
        else:
            for l in level:
                level2ss[l].append(s)
    level2s = {k: np.mean(v) for k, v in level2ss.items()}
    if do_print:
        t = sorted(list(level2s.items()), key=lambda x:x[1])
        for level, s in t:
            print(level, ' '*(20-len(level)), round(s*3.6, 2), "km/h")
    return level2s


def pre_unit(traj):
    global id2road
    edges = traj['path']
    for edge in edges:
        edge_id = edge[0]
        info = id2road[edge_id]
        geom = info['geometry']
        length = info['length']
        points = edge[1]
        points_new = []
        for point in points:
            lon, lat, tm = point['point']
            dist = geom.project(Point(lon, lat), normalized=True)
            dist *= length
            points_new.append([dist, tm])
        edge[1] = points_new
    return edges


def preprocess(traj_match, roadid_to_info):
    """预处理, 计算好每个点在道路上距离道路起点的距离"""
    result = []
    for traj in tqdm(traj_match):      # {'index': index, 'path': edges, 'start_end_portion': (start_p, end_p)}
        edges = traj['path']  
        for edge in edges:             # {'order':i, 'point':[pgeom[i][0], pgeom[i][1], times[i]], 'orig_point':orig_traj[i]}
            edge_id = edge[0]
            info = roadid_to_info[edge_id]
            geom = info['geometry']
            length = info['length']
            points = edge[1]
            points_new = []
            for point in points:
                lon, lat, tm = point['point']
                dist = geom.project(Point(lon, lat), normalized=True)
                dist *= length
                points_new.append([dist, tm])
            edge[1] = points_new
        result.append(edges)
    return result


def method_1(road):
    """一条路上有多个轨迹点，直接计算速度"""
    record = road[1]
    if len(record) < 2:
        print("point number < 2!")
        return -1

    ss = []
    total_t = 0
    for (x1, t1), (x2, t2) in zip(record, record[1:]):
        dx = x2 - x1
        dt = t2 - t1
        if dt > 0 and dx >= 0:
            s = dx / dt
            if MIN_SPEED < s < MAX_SPEED:  # 相邻两点间计算速度，排除异常数值
                ss.append((s, dt))
                total_t += dt
    if ss:
        s = sum([a * b for a, b in ss]) / total_t  # 按时长加权(若未排除异常值，相当于取首尾两点间的平均速度)
    else:
        return -1

    if s < MIN_SPEED2 and total_t < 20:  # 速度较小(11km/h)且时间长度太短
        return -1
    if s > MAX_SPEED2 and total_t < 20:  # 速度较大(100km/h)且时间长度太短
        return -1

    t = total_t / len(ss)  # 每次计算速度时，平均时间间隔
    if t < 11:
        weight = 1
    elif t < 21:
        weight = 0.8
    elif t < 31:
        weight = 0.6
    else:
        weight = 0.4

    return [s, weight]  # 估计的速度和置信度


def method_2(road, road_prev, road_next, id2road):
    """
    一条路及其相邻两条路都至少有一个轨迹点的情况:
    估计进入、离开该条路的时间，除以路径长度
    """
    if len(road[1]) == 0 or len(road_prev[1]) == 0 or len(road_next[1]) == 0:
        print("point number == 0!")
        return -1

    length = id2road[road[0]]["length"]
    length_prev = id2road[road_prev[0]]["length"]

    x_first, t_first = road[1][0]  # 该路上第一个点
    x_last, t_last = road[1][-1]  # 该路上最后一个点
    x_prev, t_prev = road_prev[1][-1]  # 上一条路上最后一个点
    x_next, t_next = road_next[1][0]  # 下一条路上第一个点

    dis1 = max(length_prev - x_prev, 0)  # 上一条路上最后一个点 与 道路起点 的距离
    dis2 = x_first  # 这条路第一个点 与 起点
    dis3 = max(length - x_last, 0)  # 这条路最后一个点 与 终点
    dis4 = x_next  # 下条路第一个点 与 终点

    # 按照时间与距离成正比的关系，估算进入、离开道路时的时间
    try:
        t_in = (dis1 * t_first + dis2 * t_prev) / (dis1 + dis2)
    except:
        t_in = (t_first + t_prev) / 2
    try:
        t_out = (dis3 * t_next + dis4 * t_last) / (dis3 + dis4)
    except:
        t_out = (t_next + t_last) / 2
    dt = t_out - t_in
    if dt <= 0:
        return -1
    s = length / dt

    # 评价结果置信度的指标为: 上一条路最后一点和这条路第一个点间隔越久，则估计越可能不准。同理这条路最后一个点和下一条路第一个点间隔。
    span = (t_first - t_prev) + (t_next - t_last)

    # 检查估计结果是否合理
    if not MIN_SPEED < s < MAX_SPEED:
        return -1
    if s < MIN_SPEED2 and span > 60:
        return -1
    if s > MAX_SPEED2 and span > 60:
        return -1

    # 根据span确定此估计结果的置信度，给出一个权重
    # span越小估计越准确，从结果观察中，具体确定区间划分
    if span < 10:
        weight = 1
    elif span < 30:
        weight = 0.8
    elif span < 50:
        weight = 0.6
    else:
        weight = 0.4

    return [s, weight]


def method_3(i, id2road, roads):
    """
    一个轨迹点在上一条路的终端附近，下一个轨迹点这条路的终端附近
    作画来说明: ______._  ______._            上一条路的终端附近，这条路的终端附近
    同理有:              _.______ _.______   这条路的首端附近，下一条路的首段附近
    同理有:    ______._  _______  _.______   上一条路的终端附近，下一条路的首端附近
    若这条路,及相邻两条路上一共至少有2个轨迹点, 则进行此估计
    """
    len_roads = len(roads)
    road = roads[i]
    results = []
    length = id2road[road[0]]["length"]

    if road[1]:
        x_orig, t_orig = road[1][0]
        x_dest, t_dest = road[1][-1]
        dis_orig = x_orig  # 离起点最近的距离
        dis_dest = max(length - x_dest, 0)  # 离终点最近的距离

        # ______._  ______._            上一条路的终端附近，这条路的终端附近
        if dis_dest < DIS_GATE or dis_dest < length / DIS_GATE2:  # 这条路的末端附近有一个点
            # 检查上条路的末端附近是否有点
            if i != 0:
                road_prev = roads[i - 1]
                if road_prev[1]:
                    length_prev = id2road[road_prev[0]]["length"]
                    x_dest_prev, t_dest_prev = road_prev[1][-1]
                    dis_dest_prev = max(length_prev - x_dest_prev, 0)
                    if (
                        dis_dest_prev < DIS_GATE
                        or dis_dest_prev < length_prev / DIS_GATE2
                    ):  # 上条路的末端附近有点
                        # 估计这条路的用时为：两点的时间差，并根据两点偏离终端的距离以假想速度修正
                        est_time = (t_dest - t_dest_prev) + (
                            dis_dest - dis_dest_prev
                        ) / FAKE_SPEED
                        s = length / est_time
                        if not (s < MIN_SPEED or s > MAX_SPEED):
                            results.append(s)

        # _.______  _.______   这条路的首端附近，下一条路的首端附近
        if dis_orig < DIS_GATE or dis_orig < length / DIS_GATE2:  # 这条路的首端附近有一个点
            # 检查下条路的首端附近是否有点
            if i != len_roads - 1:
                road_next = roads[i + 1]
                if road_next[1]:
                    length_next = id2road[road_next[0]]["length"]
                    x_orig_next, t_orig_next = road_next[1][0]
                    dis_orig_next = x_orig_next
                    if (
                        dis_orig_next < DIS_GATE
                        or dis_orig_next < length_next / DIS_GATE2
                    ):  # 下条路的首端附近有点
                        est_time = (t_orig_next - t_orig) + (
                            dis_orig - dis_orig_next
                        ) / FAKE_SPEED
                        s = length / est_time
                        if not (s < MIN_SPEED or s > MAX_SPEED):
                            results.append(s)

    # ______._  _______  _.______   上一条路的终端附近，下一条路的首端附近
    if 0 < i < len_roads - 1:
        road_prev = roads[i - 1]
        road_next = roads[i + 1]
        if road_next[1] and road_prev[1]:
            length_prev = id2road[road_prev[0]]["length"]
            length_next = id2road[road_next[0]]["length"]
            x_dest_prev, t_dest_prev = road_prev[1][-1]
            x_orig_next, t_orig_next = road_next[1][0]
            dis_dest_prev = max(length_prev - x_dest_prev, 0)
            dis_orig_next = x_orig_next
            if dis_dest_prev < DIS_GATE or dis_dest_prev < length_prev / DIS_GATE2:
                if dis_orig_next < DIS_GATE or dis_orig_next < length_next / DIS_GATE2:
                    est_time = (t_orig_next - t_dest_prev) - (
                        dis_dest_prev + dis_orig_next
                    ) / FAKE_SPEED
                    s = length / est_time
                    if not (s < MIN_SPEED or s > MAX_SPEED):
                        results.append(s)

    if results:
        return np.mean(results)
    else:
        return -1


def remove_extreme(speed):
    """使用RANSAC去除数据中的极端数据"""
    if len(speed) < 6:
        return speed
    s = np.array([spd[0] for spd in speed])
    n = round(len(s) * 0.7)
    ind = min(
        (random.sample(range(len(s)), n) for _ in range(10)), key=lambda x: s[x].std()
    )
    s = s[ind]
    a, b = min(s), max(s)
    return [i for i in speed if a <= i[0] <= b]


def speed_estimate(traj_match, id2road):
    """用三种方法从匹配轨迹估计速度"""
    speeds_by_road = {}
    method1_cnt = 0
    method2_cnt = 0
    method3_cnt = 0
    complement_cnt = 0
    for roads in tqdm(traj_match, desc="speed_estimate"):
        len_roads = len(roads)
        speed_roads = [
            {"method1": None, "method2": None, "method3": None, "complement": None}
            for i in range(len_roads)
        ]
        # 第一步
        # 遍历该条轨迹中的各条路，用3种估计速度的方法估计满足这些算法要求的路的速度
        for i in range(len_roads):
            road = roads[i]

            # 方法1: 一条路上有多个轨迹点，直接计算速度
            if len(road[1]) > 1:
                s1 = method_1(road)
                if s1 != -1:
                    speed_roads[i]["method1"] = s1
                    method1_cnt += 1

            # 方法2: 某条路及其前后两条路上都有轨迹点
            if 0 < i < len_roads - 1:
                road_prev = roads[i - 1]
                road_next = roads[i + 1]
                if len(road[1]) > 0 and len(road_prev[1]) > 0 and len(road_next[1]) > 0:
                    s2 = method_2(road, road_prev, road_next, id2road)
                    if s2 != -1:
                        speed_roads[i]["method2"] = s2
                        method2_cnt += 1

            # 方法3
            s3 = method_3(i, id2road, roads)
            if s3 != -1:
                speed_roads[i]["method3"] = [s3, 1]  # 权重设置为1
                method3_cnt += 1

        # 第二步
        # 再次考察该轨迹中的各条路，将之前没能获得速度估计、但是相邻道路都存在速度估计的道路，使用邻近道路的速度进行补充
        for i in range(len_roads):
            tmp = speed_roads[i]
            if (
                tmp["method1"] is None
                and tmp["method2"] is None
                and tmp["method3"] is None
            ):
                speeds_adj = []
                for j in range(
                    i - ADJ_GATE, i + ADJ_GATE + 1
                ):  # 使用前后各ADJ_GATE条路的所有速度估计的平均值
                    if 0 <= j < len_roads and j != i:
                        for key, value in speed_roads[j].items():
                            if key != "complement":
                                if value is not None:
                                    speeds_adj.append(value)
                if speeds_adj:
                    total_s = 0
                    total_w = 0
                    for s, w in speeds_adj:
                        total_s += s * w
                        total_w += w
                    speed_roads[i]["complement"] = [total_s / total_w, 1]  # 权重设置为1
                    complement_cnt = complement_cnt + 1

        # 保存这条轨迹产生的估计结果
        for i in range(len_roads):
            roadid = roads[i][0]
            if not roadid in speeds_by_road:
                speeds_by_road[roadid] = {
                    "method1": [],
                    "method2": [],
                    "method3": [],
                    "complement": [],
                }
            speed_est = speed_roads[i]
            for key, value in speed_est.items():
                if value is not None:
                    speeds_by_road[roadid][key].append(value)

    # 查看统计情况
    # print('method1_cnt: ', method1_cnt)
    # print('method2_cnt: ', method2_cnt)
    # print('method3_cnt: ', method3_cnt)
    # print('complement_cnt: ', complement_cnt)
    # cnt1 = 0
    # cnt2 = 0
    # cnt3 = 0
    # cnt4 = 0
    # cnt5 = 0
    # cnt6 = 0
    # for road, speed in speeds_by_road.items():
    #     speed1 = speed['method1']
    #     speed2 = speed['method2']
    #     speed3 = speed['method3']
    #     speed4 = speed['complement']
    #     if not speed1 == []:
    #         cnt1 = cnt1 + 1
    #     if not speed2 == []:
    #         cnt2 = cnt2 + 1
    #     if not speed3 == []:
    #         cnt3 = cnt3 + 1
    #     if not speed4 == []:
    #         cnt4 = cnt4 + 1
    #     if speed1 != [] or speed2 != [] or speed3 != []:
    #         cnt5 = cnt5 + 1
    #     if speed1 != [] or speed2 != [] or speed3 != [] or speed4 != []:
    #         cnt6 = cnt6 + 1
    # print('total roads exist in input trajs: ', len(speeds_by_road))
    # print('roads with speed1: ', cnt1)
    # print('roads with speed2: ', cnt2)
    # print('roads with speed3: ', cnt3)
    # print('roads with complement speed: ', cnt4)
    # print('roads with speed123: ', cnt5)
    # print('roads with speed: ', cnt6)

    return speeds_by_road


def speed_synthesize(speed_by_road, id2road):
    """综合速度估计结果"""
    # 第三步
    # 对之前产生的大量估计结果进行加权，得到每条路每种速度估计方式下的一个速度估计
    for id, speed in tqdm(speed_by_road.items(), desc="speed synthesize"):
        for key, s in speed.items():
            if s:
                s = remove_extreme(s)  # 去除数据中的极端数据
                total1 = 0
                total2 = 0
                for a, b in s:
                    total1 += a * b
                    total2 += b
                speed_by_road[id][key] = total1 / total2
    # 第四步
    # 再综合三种方式获得的速度，得到最终的速度估计
    result = {}
    for id, speed in speed_by_road.items():
        speeds = [speed["method1"], speed["method2"], speed["method3"]]
        tmp_speed = 0
        tmp_weight = 0
        for tmp, w in zip(speeds, METHOD_WEIGHT):
            if tmp:
                tmp_speed += tmp * w
                tmp_weight += w
        if tmp_weight != 0:
            result[id] = tmp_speed / tmp_weight

    eid2level = {eid: e["highway"] for eid, e in id2road.items()}
    level_to_speed = get_level2speed(eid2level, result)
    for i, j in LEVEL2SPEED.items():
        if i not in level_to_speed:
            level_to_speed[i] = j
    for id, speed in speed_by_road.items():
        if id in result:
            continue
        speeds = [speed["method1"], speed["method2"], speed["method3"]]
        tmp_speed = 0
        tmp_weight = 0
        for tmp, w in zip(speeds, METHOD_WEIGHT):
            if tmp:
                tmp_speed += tmp * w
                tmp_weight += w
        assert tmp_weight == 0
        if speed["complement"]:  # 存在补充速度, 需检查该速度是否可信
            s1 = speed["complement"]
            level = id2road[id]["highway"]
            if type(level) is list:
                s2 = np.mean([level_to_speed[x] for x in level])
            else:
                s2 = level_to_speed[level]
            if 0.7 * s2 < s1 < 1.3 * s2:
                result[id] = s1
            elif 0.5 * s2 < s1 < 1.5 * s2:
                result[id] = (s1 + s2) / 2

    print("roads with calculated speed", len(result))
    print("calculated average speed: ", np.mean(list(result.values())))
    return result


def speed_complement(G, eid2s):
    """
    根据图拓扑，对于没有速度的道路，取图中的相邻道路速度取平均值进行补充
    之后仍没有速度的道路，取道路等级相同的所有道路的平均速度
    """
    eid2s_complement = {}
    cnt = 0
    for u, v, e in G.edges(data=True):
        eid = e['id']
        if eid in eid2s:
            eid2s_complement[eid] = eid2s[eid]
            continue
        ids_adj = set()  # 找出图中这条路起点、终点上的其它所有边
        for nid in [u, v]:
            for e in list(G._pred[nid].values()) + list(G._succ[nid].values()):
                if e["id"] != eid:
                    ids_adj.add(e['id'])
        speeds_adj = []
        for id_adj in ids_adj:
            if id_adj in eid2s:
                speeds_adj.append(eid2s[id_adj])
        if len(speeds_adj) > 0: # 这条路起点、终点上的其它所有边的平均速度
            eid2s_complement[eid] = np.mean(speeds_adj)
            cnt = cnt + 1
    print('complement by topology cnt: ', cnt)

    eid2level = {e["id"]: e["highway"] for u, v, e in G.edges(data=True)}
    level2s = get_level2speed(eid2level, eid2s_complement)
    for i, j in LEVEL2SPEED.items():
        if i not in level2s:
            level2s[i] = j
    for u, v, e in G.edges(data=True):
        eid = e["id"]
        if eid not in eid2s_complement:
            level = e["highway"]
            if isinstance(level, str):
                eid2s_complement[eid] = level2s[level]
            else:
                eid2s_complement[eid] = np.mean([level2s[l] for l in level])
    print("complemented average speed: ", np.mean(list(eid2s_complement.values())))
    return eid2s_complement


def main():
    tgttop, _, config = read_config()
    G_path = f"../data_interface/G_{tgttop}.pkl"
    input_paths = [
        f"../data_interface/trajs_matched_{tgttop}_{r}.pkl" 
        for r in list(config["regions"].keys()) + ["all"]
    ]
    output_full_simple = f"../data_interface/road_speed_simple_full_{tgttop}.pkl"
    output_slice = f"../data_interface/road_speed_slice_{tgttop}.pkl"
    output_slice_simple = f"../data_interface/road_speed_slice_simple_{tgttop}.pkl"

    G = pickle.load(open(G_path, "rb"))
    global id2road
    id2road = {i["id"]: i for i in G.edges.values()}

    cache_path = f"data/traj_match_cache_{tgttop}.pkl"
    if os.path.exists(cache_path):
        traj_match = pickle.load(open(cache_path, "rb"))
    else:
        traj_match = []
        for input_path in input_paths:
            traj_match += preprocess(pickle.load(open(input_path, "rb")), id2road)
        pickle.dump(traj_match, open(cache_path, "wb"))
    print("input matched traj:", len(traj_match))

    # 不划分时间片，简单规则补全
    speeds_by_road = speed_estimate(traj_match, id2road)
    speed_est = speed_synthesize(speeds_by_road, id2road)
    speed_est = speed_complement(G, speed_est)
    pickle.dump(speed_est, open(output_full_simple, "wb"))

    # 按轨迹开始时间小时数划分24个时间片，分别估计这些时间片下的道路通行时间
    trajs_for_each_slice = defaultdict(list)
    ave_speed_for_each_slice = []
    speed_est_for_each_slice = []
    speed_est_simple_for_each_slice = []
    for traj in traj_match:
        start_tm = int(traj[0][1][0][1] / 3600)
        assert 0 <= start_tm < 24
        trajs_for_each_slice[start_tm].append(traj)
    for start_tm in range(24):
        traj_match = trajs_for_each_slice[start_tm]
        print(f"------------   start_hour:{start_tm}   ------------")
        print("input traj num:", len(traj_match))
        speeds_by_road = speed_estimate(traj_match, id2road)
        speed_est = speed_synthesize(speeds_by_road, id2road)
        ave_speed_for_each_slice.append(3.6 * np.mean(list(speed_est.values())))
        speed_est_for_each_slice.append(deepcopy(speed_est))  # 划分时间片，不补全，交给后续矩阵分解
        speed_est = speed_complement(G, speed_est)  # 划分时间片，简单规则补全
        speed_est_simple_for_each_slice.append(deepcopy(speed_est))
    print([len(x) for x in speed_est_for_each_slice])
    print([len(x) for x in speed_est_simple_for_each_slice])
    pickle.dump(
        speed_est_for_each_slice, open(output_slice, "wb")
    )
    pickle.dump(
        speed_est_simple_for_each_slice,
        open(output_slice_simple, "wb"),
    )


if __name__ == "__main__":
    main()
