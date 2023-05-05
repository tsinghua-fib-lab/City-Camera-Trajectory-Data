# -*- coding: UTF-8 -*- 
"""
对高德轨迹和商汤真值轨迹做路网匹配
输出每条数据格式:
{
    "index": index, 
    "path": [ 
        [
            eid, 
            [{"order", "point", "orig_point"}]  # 该edge上的所有轨迹点
        ], ...
    ]
    "start_end_portion": (start_p, end_p)
}
"""
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from math import ceil

import folium
import numpy as np
import osmnx as ox
from fmm import (STMATCH, GPSConfig, Network, NetworkGraph, ResultConfig,
                 STMATCHConfig)
from networkx import MultiDiGraph
from pyproj import Proj
from shapely.geometry import LineString, Point
from tqdm.contrib.concurrent import process_map
from traj_match_algo import (match_post, remove_adj_same_road_order,
                             remove_small_circle, st_match)

sys.path.append("..")
from toputils import *

workers = 16
MATCH_K = 8
MATCH_RADIUS = 0.0035
MATCH_GPS_ERROR = 0.0006
CIRCLE_SIZE = 4
CIRCLE_LENGTH = 300
ERROR_GATE = 40  # 去除平均匹配误差过大的结果
LENGTH_PORTION_GATE = 1.4  # 去除匹配后的路径明显变长的结果


def post_unit(match_result):
    """轨迹匹配结果后处理单元"""
    global nid2gps, eid2info, trajs_gps, projector
    index = match_result["index"]  # 该匹配结果 对应 输入轨迹中的第几条
    cpath = match_result["cpath"]  # 用edge_id表示的匹配路径
    cpath_id = match_result["cpath_id"]  # 每个轨迹点匹配到cpath中哪一条的索引
    orig_id = match_result["orig_id"]  # 对应原始轨迹中的第几个点（可能会删除一些点）
    error = match_result["error"]  # 每对匹配前后轨迹点间的距离
    pgeom = match_result["pgeom"]  # 每个轨迹点匹配后的坐标

    if np.mean(error) > ERROR_GATE:  # 去除平均匹配误差过大的结果
        return

    edges = cpath  # 记录edge与轨迹点的对应关系
    for i in range(len(edges)):
        edges[i] = [edges[i], []]  # 每条cpath中edge记录匹配到其上的轨迹点
    orig_traj = trajs_gps[index]
    times = [t[2] for t in orig_traj]
    for i, indice in zip(orig_id, cpath_id):
        edges[indice][1].append(
            {
                "order": i,
                "point": [pgeom[i][0], pgeom[i][1], times[i]],
                "orig_point": orig_traj[i],
            }
        )

    nodes = [eid2info[edge[0]]["od"][0] for edge in edges]
    nodes.append(eid2info[edges[-1][0]]["od"][1])
    edges = remove_small_circle(
        edges=edges,  # 去除小圈
        nodes=nodes,
        Circle_Size=CIRCLE_SIZE,
        Circle_Length=CIRCLE_LENGTH,
        nodeid_to_gps=nid2gps,
        do_not_remove_tag=False,
    )

    if len(edges) < 2:
        return

    edges = remove_adj_same_road_order(edges)  # 修复出现连续两条相同道路的情况
    for i in range(len(edges)):  # 修复首尾两条路上没有匹配的轨迹点的情况
        if edges[i][1]:
            break
    for j in range(len(edges) - 1, -1, -1):
        if edges[j][1]:
            break
    edges = edges[i : j + 1]
    if len(edges) < 2:
        return

    uvs = [eid2info[edge[0]]["od"] for edge in edges]
    last_v = uvs[0][1]
    for u, v in uvs[1:]:
        if u != last_v:  # 排除相邻道路不满足首尾相接的情况
            # print('u != last_v')
            return
        last_v = v

    # 计算首尾两条路被通过的比例
    lon, lat = edges[0][1][0]["point"][:2]
    first_edge = edges[0][0]
    geom = eid2info[first_edge]["geometry"]
    start_p = geom.project(Point(lon, lat), normalized=True)
    assert 0 <= start_p <= 1

    lon, lat = edges[-1][1][-1]["point"][:2]
    last_edge = edges[-1][0]
    geom = eid2info[last_edge]["geometry"]
    end_p = geom.project(Point(lon, lat), normalized=True)
    assert 0 <= end_p <= 1

    # 匹配轨迹路线长度/原始轨迹点连线长度
    true_length = LineString([projector(*x[:2]) for x in orig_traj]).length
    ls = [eid2info[edge[0]]["length"] for edge in edges]
    matched_length = sum(ls) - start_p * ls[0] - (1 - end_p) * ls[-1]
    if matched_length > true_length * LENGTH_PORTION_GATE:
        return
    return {"index": index, "path": edges, "start_end_portion": (start_p, end_p)}


def visualize_result(G, matched_traj, m, target):
    global trajs_gps, eid2info
    edge_geo_dict = {}
    for edge in list(G.edges(data=True)):
        xs, ys = edge[2]['geometry'].xy
        xs, ys = list(xs), list(ys)
        edge_geo_dict[edge[2]['id']] = (xs, ys)

    for t in matched_traj:
        index = t['index']
        print('plot', index)
        orig_traj = trajs_gps[index]
        t = t['path']
        cpath, points = [], []
        for x in t:
            cpath.append(x[0])
            points.extend(x[1])
        points.sort(key=lambda x: x['order'])
        matched_points = [x['point'] for x in points]
        orig_points = [x['orig_point'] for x in points]
        if orig_points != orig_traj:
            for x in orig_points:
                assert x in orig_traj
            print('removed points:', len(orig_traj) - len(orig_points))

        nodes = [eid2info[edge]["od"][0] for edge in cpath]
        nodes.append(eid2info[cpath[-1]]["od"][1])
        G_folium = ox.folium.plot_graph_folium(G, graph_map=m, edge_color='gray', edge_width=3, edge_opacity=0.5)
        locations = []
        for lon, lat, tm in orig_traj:
            folium.CircleMarker(location=[lat, lon], radius=6, color='black').add_to(G_folium)  # 原始轨迹点
            locations.append((lat, lon))
        folium.PolyLine(locations=locations, color='black', weight=4).add_to(G_folium)          # 原始轨迹折线
        for lon, lat, tm in matched_points:
            folium.CircleMarker(location=[lat, lon], radius=4, color='red', popup=(lon, lat, tm)).add_to(G_folium)    # 匹配轨迹点
        # G_folium = ox.plot_route_folium(G, route=nodes, route_map=G_folium)
        xs, ys = [], []
        for edge in cpath:
            x, y = edge_geo_dict[edge]
            xs.extend(x)
            ys.extend(y)
        locations = [(y, x) for x, y in zip(xs, ys)]
        folium.PolyLine(locations=locations, color='red', weight=4).add_to(G_folium)  # 匹配路径
        G_folium.save(f'figure/match_result_{index}_{target}.html')


def main():
    global trajs_gps, projector, nid2gps, eid2info

    tgttop, _, config, projector = read_config(proj=True)
    G_all = pickle.load(open(f"../data_interface/G_{tgttop}.pkl", "rb"))
    r2G = pickle.load(open(f"../data_interface/r2G_{tgttop}.pkl", 'rb'))
    r2G["all"] = G_all  # 各区域外的轨迹, 匹配到大路网

    # 估计1m距离对应的经纬度gps_1m
    lon, lat = json.loads(config["center"])
    x, y = projector(lon, lat)
    lon1, _ = projector(x+1, y, inverse=True)
    _, lat1 = projector(x, y+1, inverse=True)
    gps_1m = ((lon1 - lon) + (lat1 - lat)) / 2
    print("gps_1m:", gps_1m)
    
    for r in list(config["regions"].keys()) + ["all"]:  # 匹配高德
    # for r in ["all"]:  # 匹配真值
        print("region:", r)

        input_path = f"data/trajs_{tgttop}_{r}.pkl"
        traj_csv_path = f"data/trajs_csv_{tgttop}_{r}.csv"    # FMM读gps轨迹需要写成csv文件的形式
        mm_result_path = f"data/mm_result_{tgttop}_{r}.txt"   # FMM输出的txt文件
        output_path = f"../data_interface/trajs_matched_{tgttop}_{r}.pkl"
        
        # input_path = f"data/gt_vid2trajs_{tgttop}.pkl"
        # traj_csv_path = f"data/gt_trajs_csv_{tgttop}.csv"   # FMM读gps轨迹需要写成csv文件的形式
        # mm_result_path = f"data/gt_mm_result_{tgttop}.txt"  # FMM输出的txt文件
        # output_path = f"../data_interface/gt_vid2trajs_matched_{tgttop}.pkl"
        
        G = MultiDiGraph(r2G[r])  # FMM需要MultiDiGraph
        nid2gps = {nid:(n["x"], n["y"]) for nid, n in G.nodes(data=True)}
        eid2info = {e["id"]: e for u, v, e in G.edges(data=True)}
        shp_path = f"data/shp_files_{tgttop}_{r}"          # FMM需要把G转成shapefile

        trajs_gps = pickle.load(open(input_path, "rb"))
        # vid2trajs = pickle.load(open(input_path, "rb"))
        # vid_trajs = [(vid, traj) for vid, trajs in vid2trajs.items() for traj in trajs]
        # vids, trajs_gps = zip(*vid_trajs)
        print("trajs_gps:", len(trajs_gps))

        # 调用map-match算法
        if not os.path.exists(mm_result_path):
            st_match(
                G=G, 
                traj_data=trajs_gps, 
                config={"k": MATCH_K, "radius": MATCH_RADIUS, "gps_error": MATCH_GPS_ERROR},
                traj_csv_path=traj_csv_path,
                shp_path=shp_path,
                mm_result_path=mm_result_path)
            
        # 通用后处理
        match_result = match_post(
            G=G, 
            mm_result_path=mm_result_path, 
            gps_1m=gps_1m)

        # 后处理, 将匹配前的轨迹点与匹配结果进行对应
        batch_sz = 10000  # 避免内存占用过大
        trajs_matched = []
        for i in range(0, len(match_result), batch_sz):
            batch = match_result[i: i + batch_sz]
            result = process_map(
                post_unit,
                batch,
                chunksize=min(ceil(len(batch) / workers), 500),
                max_workers=workers,
            )
            trajs_matched += [i for i in result if i]
        print("trajs_matched:", len(trajs_matched))

        pickle.dump(trajs_matched, open(output_path, "wb"))  # {'index': index, 'path': edges, 'start_end_portion': (start_p, end_p)}
        # m = get_base_map()
        # visualize_result(G, random.sample(trajs_matched, 20), m, tgttop)

        # vid2trajs_matched = defaultdict(list)
        # for tj in trajs_matched:
        #     vid2trajs_matched[vids[tj["index"]]].append(tj)
        # pickle.dump(vid2trajs_matched, open(output_path, "wb"))
        # m = get_base_map()
        # trajs = random.sample(list(vid2trajs_matched.values()), 10)
        # # visualize_result(G, [x[0] for x in trajs], m, tgttop + "gt")
        

if __name__ == "__main__":
    main()
