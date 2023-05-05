"""
node:
    {
        "x": lon,
        "y": lat,
        "xx": x,
        "yy": y,
        "camera": []  # optional
    }
edge:
    {
        "id":
        "points_gps": [(lon, lat), ...],
        "points_xy": [(x, y), ...],
        "geometry": LineString(points_gps),
        "highway": level,
        "od": [uid, vid],
        "length":
        其它未使用字段
    }
camera:
    {
        "id":
        "name":
        "node_id":
        "gps": node_gps,
        "gps_orig": camera_orig_gps
    }
"""
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from math import atan2, ceil, pi as PI
from pprint import pprint

import folium
import numpy as np
import osmnx as ox
from coord_convert.transform import wgs2gcj
from networkx import DiGraph, MultiDiGraph
from shapely.geometry import LineString, Point, Polygon
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

sys.path.append("..")
from toputils import *

tgttop, _, config, projector = read_config(proj=True)

input_camera_path = f"data/r2cameras_{tgttop}.pkl"
output_camera_path = f"../data_interface/r2cameras_{tgttop}.pkl"
output_map_path = f"../data_interface/G_{tgttop}.pkl"
output_region2map_path = f"../data_interface/r2G_{tgttop}.pkl"

MERGE_NODE_GATE = 20          # 邻近node合并距离阈值
MATCH_CAMERA_NODE_GATE = 20   # 摄像头匹配到路口距离阈值
MATCH_CAMERA_EDGE_GATE = 20   # 摄像头匹配到路中距离阈值
SQRT2 = 2 ** 0.5
PI2 = PI * 2
global lines_matcher
workers = 1


def plot_graph(G, cameras=[], cameras_new=[], path="figure/test.html", save=True):
    m = get_base_map(gaode=True)
    for nid, n in G.nodes(data=True):
        folium.CircleMarker(
            location=wgs2gcj(n["x"], n["y"])[::-1],
            radius=2,
            color="black",
            opacity=0.5,
            popup=nid
        ).add_to(m)
    if isinstance(G, MultiDiGraph):
        for u, v, k in G.edges:
            e = G.edges[u, v, k]
            folium.PolyLine(
                locations=[wgs2gcj(*p)[::-1] for p in e["points_gps"]],
                weight=2,
                color="gray",
                opacity=0.5,
                popup=(u, v, k, e["highway"])
            ).add_to(m)
    else:
        for u, v, e in G.edges(data=True):
            folium.PolyLine(
                locations=[wgs2gcj(*p)[::-1] for p in e["points_gps"]],
                weight=2,
                color="gray",
                opacity=0.5,
                popup=(u, v, e["highway"])
            ).add_to(m)
    for c in cameras_new:
        folium.CircleMarker(
            location=wgs2gcj(*c["gps"])[::-1],
            radius=3,
            popup=(c["id"], c["node_id"], c["name"]),
            color="blue"
        ).add_to(m)
        folium.CircleMarker(
            location=wgs2gcj(*c["gps_orig"])[::-1],
            radius=3,
            popup=(c["id"], c["node_id"], c["name"]),
            color="orange"
        ).add_to(m)
        ps = [c["gps_orig"], c["gps"]]
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in ps],
            popup=(c["id"], c["node_id"], c["name"]),
            weight=2
        ).add_to(m)
        for ps_roads in c["gps_roads"].values():
            for p in ps_roads:
                if p != c["gps"]:
                    ps = [c["gps_orig"], p]
                    folium.PolyLine(
                        locations=[wgs2gcj(*p)[::-1] for p in ps],
                        popup=(c["id"], c["node_id"], c["name"]),
                        weight=2
                    ).add_to(m)
    cids = {c["id"] for c in cameras_new}
    for c in cameras:
        if c["id"] not in cids:
            folium.CircleMarker(
                location=wgs2gcj(*c["gps"])[::-1],
                radius=3,
                popup=(c["id"], c["name"]),
                color="red"
            ).add_to(m)
    if save:
        m.save(path)
    return m


def get_graph(bound_poly):
    # 下载osm数据
    cache_path = f"data/osm_{tgttop}.pkl"
    if not os.path.exists(cache_path):
        G = ox.graph.graph_from_polygon(
            polygon=bound_poly,
            network_type="drive", 
            simplify=True, 
            retain_all=False, 
            truncate_by_edge=True, 
            clean_periphery=True, 
            custom_filter=None)
        pickle.dump(G, open(cache_path, "wb"))
    else:
        print("Loading osm cache")
        G = pickle.load(open(cache_path, "rb"))
    # print(type(G))   # "networkx.classes.multidigraph.MultiDiGraph"
    print("nodes num: ", len(G.nodes))  # 21408
    print("edges num: ", len(G.edges))  # 44200

    # 合并路口节点
    print("Merging nodes...")
    G_proj = ox.project_graph(G, to_crs=config["crs"])
    G = ox.consolidate_intersections(
        G=G_proj, 
        tolerance=MERGE_NODE_GATE, 
        rebuild_graph=True, 
        dead_ends=False, 
        reconnect_edges=True)
    print("nodes num: ", len(G.nodes))  # 2251
    print("edges num: ", len(G.edges))  # 5364

    # 将node的属性统一为{"x": lon, "y": lat, "xx": x, "yy": y}
    for node in G.nodes(data=True):
        nid = node[0]
        data = node[1]
        x, y = data["x"], data["y"]
        if "lon" in data:
            lon, lat = data["lon"], data["lat"]
            x1, y1 = projector(lon, lat)
            assert (x - x1) ** 2 + (y - y1) ** 2 < 1
        lon, lat = projector(x, y, inverse=True)
        keys = list(node[1].keys())
        for key in keys:  # 不允许直接给G.nodes[id]赋值新的dict
            G.nodes[nid].pop(key, None)
        G.nodes[nid]["x"] = lon
        G.nodes[nid]["y"] = lat
        G.nodes[nid]["xx"] = x
        G.nodes[nid]["yy"] = y

    # 将edge赋予自然id，去除无用信息，重新计算道路长度，增加OD及其坐标信息
    level_set = {'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 
        'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'living_street', 'residential'}
    for uid, (u, v, k) in enumerate(G.edges):
        for key in ["u_original", "v_original", "osmid"]:
            G.edges[u, v, k].pop(key, None)  # 去除无用信息
        G.edges[u, v, k]["id"] = uid         # 赋予唯一id

        info = G.edges[u,v,k]
        level = info["highway"]
        if isinstance(level, str):
            if level not in level_set:
                print("unexpected level:", level)
                info["highway"] = "unclassified"
        else:
            info["highway"] = [l if l in level_set else "unclassified" for l in level]
            
        xys = list(info["geometry"].coords)  # 经过合并节点操作后，图中的信息都变成了平面坐标，为了后续应用，改回经纬度坐标
        gpss = [projector(*p, inverse=True) for p in xys]
        gpss[0] = (G.nodes[u]["x"], G.nodes[u]["y"])     # 确保geometry的OD和uv坐标完全相同
        gpss[-1] = (G.nodes[v]["x"], G.nodes[v]["y"])
        G.edges[u, v, k]["geometry"] = LineString(gpss)
        xys = [projector(*p) for p in gpss]
        length = LineString(xys).length
        if abs(info["length"] - length) > 10:
            print("length changed a lot:", round(info["length"]), round(length))
        G.edges[u, v, k]["length"] = length     # 重算length
        G.edges[u, v, k]["points_gps"] = gpss   # 记录形状点
        G.edges[u, v, k]["points_xy"] = xys
        
        G.edges[u, v, k]["od"] = (u, v)  # 记录od

    # 修复k不从0开始的问题
    nodes = list(G.nodes(data=True))
    edges = [(u, v, info) for u, v, info in G.edges(data=True)]
    G = MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    edge_ids = [edge[2]["id"] for edge in edges]
    assert len(edge_ids) == len(set(edge_ids))
    node_ids = [node[0] for node in nodes]
    assert len(node_ids) == len(set(node_ids))

    return G


def remove_multiple_edge(G):
    """处理同一u, v有多个k的路, 保留等级高、长度短的那条"""
    def level_higer_than(edge1, edge2):
        t1 = edge1['highway']
        t2 = edge2['highway']
        if t1 == t2:
            return False
        ranked_levels = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary',
                         'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'living_street', 'residential']
        if isinstance(t1, str):
            rank1 = ranked_levels.index(t1)
        else:
            rank1 = min([ranked_levels.index(t) for t in t1])
        if isinstance(t2, str):
            rank2 = ranked_levels.index(t2)
        else:
            rank2 = min([ranked_levels.index(t) for t in t2])
        if rank1 == rank2:
            return edge1["length"] < edge2["length"]  # level相同时, length越小越好
        else:
            return rank1 < rank2

    uv2ks = defaultdict(list)
    for u, v, k in G.edges:
        uv2ks[(u, v)].append(k)
    uv2ks = {(u, v): ks for (u, v), ks in uv2ks.items() if len(ks) > 1}
    # m = plot_graph(G, save=False)
    remove_cnt = 0
    for u, v in uv2ks:
        edges = list(G[u][v].items())
        # for k, edge in edges:
        #     folium.PolyLine(
        #         locations=[wgs2gcj(*p)[::-1] for p in edge['geometry'].coords], 
        #         color='red', weight=3).add_to(m)

        to_be_del = []
        while len(edges) > 1:
            removed_flag = False
            for i1 in range(len(edges)):
                k1, edge1 = edges[i1]
                # geo1 = edge1['geometry']
                for i2 in range(i1+1, len(edges)):
                    k2, edge2 = edges[i2]
                    if level_higer_than(edge1, edge2):
                        to_be_del.append(k2)            # 去掉等级较低的那条
                        edges.pop(i2)
                    else:
                        to_be_del.append(k1)
                        edges.pop(i1)
                    removed_flag = True                 # 去掉一条后重新开始做两重for循环
                    break
                if removed_flag:
                    break
            if not removed_flag:                        # 遍历每对edge，均不合并，则退出
                break
        assert len(edges) > 0

        for k in to_be_del:
            # t = G.edges[u,v,k]
            # folium.PolyLine(
            #     locations=[wgs2gcj(*p)[::-1] for p in t['geometry'].coords], 
            #     color='blue', weight=3).add_to(m)
            G.remove_edge(u, v, k)
        remove_cnt += len(to_be_del)
    
    # m.save("figure/test.html")
    print('removed edges:', remove_cnt)

    uv2ks = defaultdict(list)
    for u, v, k in G.edges:
        uv2ks[(u, v)].append(k)
    assert all(len(ks) == 1 for ks in uv2ks.values())

    # 转为DiGraph
    nodes = list(G.nodes(data=True))
    edges = [(u, v, info) for u, v, info in G.edges(data=True)]
    G = DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


def match_unit(camera):
    """
    未能匹配到node的camera, 匹配到edge
    同一camera可同时匹配到一定距离阈值范围内的多条edge
    """
    global lines_matcher
    cid = camera["id"]
    geo = camera["geo"]
    x, y = camera["point"]
    dis_gate = MATCH_CAMERA_EDGE_GATE

    candidates = []
    double_dis_gate = dis_gate * 2
    for line in lines_matcher:
        x2, y2 = line["point"]
        # if (abs(x - x2) + abs(y - y2)) / SQRT2 < lane['length'] / 2 + dis_gate:
        tmp = SQRT2 * (abs(x - x2) + abs(y - y2)) - line["length"]  # 先根据1范数估计距离的上限
        if tmp < double_dis_gate:
            s = line["geo"].project(geo, normalized=True)
            if 0 < s < 1:  # 不允许匹配到路的端点, 否则早已应被匹配到node
                point2 = line["geo"].interpolate(s, normalized=True)
                distance = geo.distance(point2)
                if distance < dis_gate:
                    candidates.append([distance, line["id"], s, point2])
    if not candidates:
        return cid   # 匹配失败
    return(
        cid,
        [
            (eid, s, projector(*point2.coords[:][0], inverse=True))
            for _, eid, s, point2 in candidates  # 应该允许camera同时匹配到多条路上(例如两条平行的反向路)
        ]
    )


def calc_dis(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calc_angle(ps, pe):
    xs, ys = ps
    xe, ye = pe
    return atan2(ye - ys, xe - xs)


def norm_angle(angle):
    return (angle + PI) % PI2 - PI


def match_cam_to_roads(ris, ros, c):
    """对部署在路口处的摄像头, 将其匹配到路上"""
    ris = {r["id"]: deepcopy(r) for r in ris}
    ros = {r["id"]: deepcopy(r) for r in ros}
    p, p_orig = projector(*c["gps"]), projector(*c["gps_orig"])
    a = calc_angle(p, p_orig)
    in_roads = set()
    out_roads = set()
    # 计算一些基本属性
    for n, rs in enumerate([ris, ros]):
        for r in rs.values():
            ps = r["points_xy"]
            if n == 0:
                ps = ps[::-1]
            r["line"] = LineString(ps)
            if len(ps) > 2 and calc_dis(ps[0], ps[1]) < 10:
                ps = ps[1:]  # 去掉被吸入路口内而拐弯的一段
            r["angle"] = calc_angle(ps[0], ps[1])
            r["parallel"] = set()
    # 找同一方向的入路、出路对
    for i, ri in ris.items():
        ai = ri["angle"]
        for j, ro in ros.items():
            if abs(norm_angle(ai - ro["angle"])) < PI / 8:
                ri["parallel"].add(j)
                ro["parallel"].add(i)
    # for x in [ris, ros]:
    #     for i, r in x.items():
    #         print(i, round(r["angle"] / PI * 180), r["parallel"])
    # 若从路口指向摄像头位置的方向, 与路的方向基本相同, 则匹配
    succeed_flag = False
    for n, rs in enumerate([ris, ros]):
        adif, i = min([
            [
                abs(norm_angle(a - r["angle"])),
                i
            ] for i, r in rs.items()
        ])
        if adif < PI / 16:
            succeed_flag = True
            if n == 0:
                in_roads.add(i)
                out_roads |= ris[i]["parallel"]
            else:
                out_roads.add(i)
                in_roads |= ros[i]["parallel"]
    if succeed_flag:
        return {"in": list(in_roads), "out": list(out_roads)}
    # 通过方向匹配失败, 则匹配到投影距离最近的路
    p_orig = Point(p_orig)
    dis_n_i = []
    for n, rs in enumerate([ris, ros]):
        for i, r in rs.items():
            dis = p_orig.distance(r["line"].interpolate(r["line"].project(p_orig)))
            dis_n_i.append([dis, n, i])
    dis, n, i = min(dis_n_i, key=lambda x: x[0])
    if n == 0:
        in_roads.add(i)
        out_roads |= ris[i]["parallel"]
    else:
        out_roads.add(i)
        in_roads |= ros[i]["parallel"]
    return {"in": list(in_roads), "out": list(out_roads)}


def match_camera_to_graph(G, r2cameras, workers=16):
    """
    将摄像头匹配到路网
    优先匹配到一定距离内的node
    其次匹配到一定距离内的edge并将路网切分开
    匹配不上的摄像头去掉
    """
    global lines_matcher
    r2cids = {r: set(c["id"] for c in cams) for r, cams in r2cameras.items()}
    cameras = list({c["id"]: c for cams in r2cameras.values() for c in cams}.values())  # 去除辅助区域与聚类区域重合部分的重复camera
    cameras_new = []

    # 匹配到node
    print("Matching cameras to nodes...")
    nid2cids = defaultdict(list)
    nodes = list(G.nodes(data=True))
    dis_gate2 = MATCH_CAMERA_NODE_GATE ** 2
    cameras_undecided = []
    for c in tqdm(cameras):
        lon, lat = c["gps"]
        x, y = projector(lon, lat)
        c["xy"] = (x, y)
        for nid, n in nodes:
            x2, y2 = n["xx"], n["yy"]
            if (x - x2) ** 2 + (y - y2) ** 2 < dis_gate2:
                nid2cids[nid].append(c["id"])
                cameras_new.append({
                    "id": c["id"],
                    "node_id": nid,
                    "name": c["name"],
                    "gps": (n["x"], n["y"]),
                    "gps_orig": c["gps"]
                })
                break
        else:
            cameras_undecided.append(c)
    print("cameras matched to nodes:", len(cameras) - len(cameras_undecided))
    print("cameras undecided:", len(cameras_undecided))

    # 匹配到edge
    print("Matching cameras to edges...")
    lines_matcher = [
        (LineString(e["points_xy"]), e["id"])
        for _, _, e in G.edges(data=True)
    ]
    lines_matcher = [{
        "id": eid,
        "geo": geo,
        "point": geo.interpolate(0.5, normalized=True).coords[:][0],  # 折线的中点
        "length": geo.length,
        } for geo, eid in lines_matcher
    ]
    cameras_matcher = [{
        "id": c["id"],
        "geo": Point(c["xy"]),
        "point": c["xy"]
        } for c in cameras_undecided
    ]
    results = process_map(
        match_unit, 
        cameras_matcher, 
        chunksize=min(ceil(len(cameras_matcher) / workers), 100), 
        max_workers=workers)
    cids_not_matched = [x for x in results if not isinstance(x, tuple)]
    results = [x for x in results if isinstance(x, tuple)]
    print("cameras matched to edges:", len(results))
    print("cameras not matched:", len(cids_not_matched))

    # 新增投影点为路网节点
    cameras = {c["id"]: c for c in cameras}
    eid2edge = {e["id"]: e for u, v, e in G.edges(data=True)}
    new_nodes = []
    new_gps2nid = {}  # 当不同摄像头投影到坐标完全相同的点时, 不要新建多个node
    eid2s_nid = defaultdict(list)
    nid = max(G.nodes) + 1
    for cid, eid_s_gps in results:
        c = cameras[cid]
        if len(eid_s_gps) == 1:  # 摄像头只投影到1条路上
            eid, s, gps = eid_s_gps[0]
            xy = projector(*gps)
            if gps not in new_gps2nid:
                new_nodes.append((nid, {"x": gps[0], "y": gps[1], "xx": xy[0], "yy": xy[1]}))
                new_gps2nid[gps] = nid
                eid2s_nid[eid].append((s, nid))
                nid += 1
            cameras_new.append({
                "id": c["id"],
                "node_id": new_gps2nid[gps],
                "name": c["name"],
                "gps": (gps[0], gps[1]),
                "gps_orig": c["gps"]
            })
        else:
            xys = [projector(*x[-1]) for x in eid_s_gps]
            xs, ys = zip(*xys)
            x, y = np.mean(xs), np.mean(ys)  # 投影到多条路上时, 取平均坐标
            lon, lat = projector(x, y, inverse=True)
            gps = (lon, lat)
            point = Point((x, y))
            is_matched_to_node = False
            if gps not in new_gps2nid:
                for eid, _, _ in eid_s_gps:
                    edge = eid2edge[eid]
                    line = LineString(edge["points_xy"])
                    s = line.project(point, normalized=True)
                    if not 0 < s < 1:  # 因为以上取平均坐标的操作, 有可能投影到端点, 则按匹配到路口的情况处理
                        is_matched_to_node = True
                        tmp = edge["od"][0] if s == 0 else edge["od"][1]
                        n = G.nodes[tmp]
                        cameras_new.append({
                            "id": c["id"],
                            "node_id": tmp,
                            "name": c["name"],
                            "gps": (n["x"], n["y"]),
                            "gps_orig": c["gps"]
                        })
                        break
                else:  # 投影点确实位于路中间
                    new_nodes.append((nid, {"x": lon, "y": lat, "xx": x, "yy": y}))
                    new_gps2nid[gps] = nid
                    for eid, _, _ in eid_s_gps:
                        edge = eid2edge[eid]
                        line = LineString(edge["points_xy"])
                        s = line.project(point, normalized=True)
                        assert 0 < s < 1
                        eid2s_nid[eid].append((s, nid))
                    nid += 1
            if not is_matched_to_node:
                cameras_new.append({
                    "id": c["id"],
                    "node_id": new_gps2nid[gps],
                    "name": c["name"],
                    "gps": (lon, lat),
                    "gps_orig": c["gps"]
                })
    print("new_nodes:", len(new_nodes))
    assert len(cameras) - len(cameras_new) == len(cids_not_matched)

    # 在投影点处拆分道路
    new_edges = []
    eid_new = max(eid2edge.keys()) + 1
    nid2node = {nid:n for nid, n in new_nodes}
    for eid, s_nids in tqdm(eid2s_nid.items()):
        # 计算路上原本的中间点的s
        e = eid2edge[eid]
        del e["geometry"]
        gpss = e["points_gps"]
        xys = e["points_xy"]
        line = LineString(xys)
        s_gpss = []
        for gps, xy in zip(gpss[1:-1], xys[1:-1]):
            s_gpss.append((line.project(Point(xy), normalized=True), gps))
        # 和匹配点放在一起, 排序
        s_nids_gpss = s_nids + s_gpss
        s_nids_gpss.sort(key=lambda x:x[0])
        o, path = e["od"][0], [gpss[0]]
        for _, nid_or_gps in s_nids_gpss:
            if isinstance(nid_or_gps, tuple) or isinstance(nid_or_gps, list):  # gps
                path.append(nid_or_gps)
            else:
                n = nid2node[nid_or_gps]
                path.append((n["x"], n["y"]))
                info = deepcopy(e)
                info["id"] = eid_new
                info["od"] = (o, nid_or_gps)
                info["points_gps"] = path
                info["points_xy"] = [projector(*p) for p in path]
                info["length"] = LineString(info["points_xy"]).length
                new_edges.append((o, nid_or_gps, info))
                o = nid_or_gps
                path = [(n["x"], n["y"])]
                eid_new += 1
        info = deepcopy(e)
        info["id"] = eid_new
        info["od"] = (o, e["od"][1])
        info["points_gps"] = path + [gpss[-1]]
        info["points_xy"] = [projector(*p) for p in info["points_gps"]]
        info["length"] = LineString(info["points_xy"]).length
        new_edges.append((o, e["od"][1], info))
        eid_new += 1
    print("new_edges:", len(new_edges))
    
    # 去除拆分前的边
    to_remove = []
    for u, v, e in G.edges(data=True):
        if e["id"] in eid2s_nid:
            to_remove.append((u, v))
    print("remove edges:", len(to_remove))
    for u, v in to_remove:
        G.remove_edge(u, v)

    # 添加新点和边
    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)

    # 节点添加camera字段
    for c in cameras_new:
        n = G.nodes[c["node_id"]]
        if "camera" not in n:
            n["camera"] = [c["id"]]
        else:
            n["camera"].append(c["id"])
    
    # 将摄像头匹配到路
    new_nids = set(new_gps2nid.values())
    eid2edge = {e["id"]: e for u, v, e in G.edges(data=True)}
    for c in tqdm(cameras_new, desc="match cam to roads"):
        nid = c["node_id"]
        in_roads = list(G._pred[nid].values())   # 前驱路
        out_roads = list(G._succ[nid].values())  # 后继路
        if c["node_id"] in new_nids:  # 匹配到道路中间的摄像头
            c["roads"] = {
                "in": [e["id"] for e in in_roads], 
                "out": [e["id"] for e in out_roads]
            }
            c["gps_roads"] = {
                "in": [c["gps"] for _ in range(len(in_roads))], 
                "out": [c["gps"] for _ in range(len(out_roads))]}
        else:  # 匹配到路口的摄像头
            c["roads"] = match_cam_to_roads(in_roads, out_roads, c)
            ps_in, ps_out = [], []  # 求投影到路上的投影点坐标
            p = Point(projector(*c["gps_orig"]))
            for eids, ps in zip([c["roads"]["in"], c["roads"]["out"]], [ps_in, ps_out]):
                for eid in eids:
                    line = LineString(eid2edge[eid]["points_xy"])
                    p_proj = line.interpolate(line.project(p)).coords[:][0]
                    ps.append(projector(*p_proj, inverse=True))
            c["gps_roads"] = {"in": ps_in, "out": ps_out}

    # 检查结果合法
    assert len(cameras_new) == len(cameras) - len(cids_not_matched)
    cid2nid = {c["id"]: c["node_id"] for c in cameras_new}
    nid2cids = {nid: n["camera"] for nid, n in G.nodes.items() if "camera" in n}
    for cid, nid in cid2nid.items():
        assert cid in nid2cids[nid]
    for nid, cids in nid2cids.items():
        for cid in cids:
            assert cid2nid[cid] == nid

    r2cameras_new = defaultdict(list)
    for c in cameras_new:
        for r, cids in r2cids.items():
            if c["id"] in cids:
                r2cameras_new[r].append(c)

    print("matched for each region")
    pprint([
       (r, len(cams), len(r2cameras_new[r]))
       for r, cams in r2cameras.items()
    ])

    return G, r2cameras_new


def post_process(G):
    # 去除自环
    to_remove = []
    for u, v in G.edges:
        if u == v:
            to_remove.append((u, v))
    print("remove self ring:", len(to_remove))
    for u, v in to_remove:
        G.remove_edge(u, v)

    # 坐标保持精确自洽, 避免浮点计算带来的微小不同
    for nid, n in G.nodes(data=True):
        lon, lat = n["x"], n["y"]
        x, y = n["xx"], n["yy"]
        x2, y2 = projector(lon, lat)
        assert (x - x2) ** 2 + (y - y2) ** 2 < 1
        n["xx"], n["yy"] = x2, y2
    for u, v, e in G.edges(data=True):
        assert (u, v) == e["od"]
        gpss = e["points_gps"]
        assert gpss[0] == (G.nodes[u]["x"], G.nodes[u]["y"])
        assert gpss[-1] == (G.nodes[v]["x"], G.nodes[v]["y"])
        xys = e["points_xy"]
        xys[0] = (G.nodes[u]["xx"], G.nodes[u]["yy"])
        xys[-1] = (G.nodes[v]["xx"], G.nodes[v]["yy"])
        e["points_xy"] = xys
        e["geometry"] = LineString(gpss)
        e["length"] = LineString(xys).length
        assert e["length"] > 0

    # 检查id不重复
    node_ids = [node[0] for node in G.nodes(data=True)]
    assert len(node_ids) == len(set(node_ids))
    eids = [e["id"] for u, v, e in G.edges(data=True)]
    assert len(eids) == len(set(eids))
    
    # node添加succ, pred字段
    for nid, n in G.nodes(data=True):
        n["succ"], n["pred"] = [], []
    for u, v, e in G.edges(data=True):
        G.nodes[u]["succ"].append(e["id"])
        G.nodes[v]["pred"].append(e["id"])

    # 添加crs信息
    G.graph["crs"] = config["crs"]

    # 切分出每个区域的小路网: 至少有1个点在区域内的路及其端点
    print("Cutting G for each region...")
    poly_rs = [
        [
            Polygon([projector(*p) for p in json.loads(v["bound"])]), 
            r
        ] 
        for r, v in config["regions"].items()
    ]
    nodes = {x[0]: x for x in list(G.nodes(data=True))}
    edges = {x[-1]["id"]: x for x in list(G.edges(data=True))}
    r2nids = defaultdict(set)
    r2eids = defaultdict(set)
    for nid, n in tqdm(nodes.items()):
        p = Point(n[1]["xx"], n[1]["yy"])
        for poly, r in poly_rs:
            if p.covered_by(poly):
                r2nids[r].add(nid)
    r2nids_new = defaultdict(set)
    for eid, (u, v, e) in edges.items():
        for r, nids in r2nids.items():
            a, b = u in nids, v in nids
            if a or b:
                r2eids[r].add(eid)
                if not a:
                    r2nids_new[r].add(u)
                if not b:
                    r2nids_new[r].add(v)

    r2nids = {r: nids | r2nids_new[r] for r, nids in r2nids.items()}
    r2G = {}
    for r, nids in r2nids.items():
        eids = r2eids[r]
        G_r = DiGraph()
        G_r.add_nodes_from([nodes[nid] for nid in nids])
        G_r.add_edges_from([edges[eid] for eid in eids])
        G_r.graph["crs"] = config["crs"]
        r2G[r] = G_r

    return G, r2G


if __name__ == "__main__":
    cache_path = f"data/G_orig_{tgttop}.pkl"
    if not os.path.exists(cache_path):
        G = get_graph(Polygon(json.loads(config["bound"])))
        pickle.dump(G, open(cache_path, "wb"))
    else:
        G = pickle.load(open(cache_path, "rb"))

    # plot_graph(G, path=f"figure/G_orig_{tgttop}.html")
    
    cache_path = f"data/G_simplified_{tgttop}.pkl"
    if not os.path.exists(cache_path):
        G = remove_multiple_edge(G)
        pickle.dump(G, open(cache_path, "wb"))
    else:
        G = pickle.load(open(cache_path, "rb"))

    # plot_graph(G, path=f"figure/G_simplified_{target}.html")

    r2cameras = pickle.load(open(input_camera_path, "rb"))
    G, r2cameras_new = match_camera_to_graph(G, r2cameras, workers)
    pickle.dump(r2cameras_new, open(output_camera_path, "wb"))

    G, r2G = post_process(G)
    pickle.dump(G, open(output_map_path, "wb"))
    pickle.dump(r2G, open(output_region2map_path, "wb"))

    cameras = list({c["id"]: c for cams in r2cameras.values() for c in cams}.values())
    cameras_new = list({c["id"]: c for cams in r2cameras_new.values() for c in cams}.values())
    plot_graph(G, cameras, cameras_new, path=f"figure/G_add_cameras_{tgttop}.html")
    # for r, G_r in r2G.items():
    #     plot_graph(G_r, r2cameras[r], r2cameras_new[r], path=f"figure/G_add_cameras_{tgttop}_{r}.html")

    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))
    print("first node:")
    pprint(nodes[0])
    print("first edge:")
    pprint(edges[0])

    print("road level distribution:")
    levels = []
    for edge in edges:
        level = edge[2]["highway"]
        if isinstance(level, str):
            levels.append(level)
        else:
            levels.extend(level)
    levels = list(dict(Counter(levels)).items())
    levels.sort(key=lambda x: x[1], reverse=True)
    for level, num in levels:
        print(level, " "*(20-len(level)), num)
