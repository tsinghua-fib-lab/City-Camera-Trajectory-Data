import pickle
import random
import sys
from collections import defaultdict
from math import ceil
import os
import folium
import networkx as nx
import yaml
from coord_convert.transform import wgs2gcj
from main import cut_distant_points, merge_tm_adj_points
from routing import Router
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

sys.path.append("..")
from toputils import *

random.seed(233)
ADJ_RANGE = 180
TM_GAP_GATE = 1800
workers = 8

tgttop, target, config, projector = read_config(proj=True)

def get_records_labels(target, cameras):
    cfg = yaml.load(open(config["regions"][target]["main_config"], "r", encoding="utf8"), Loader=yaml.SafeLoader)
    exp_name = cfg["name"]
    records = pickle.load(open(f"../data_interface/records_pca_{config['pca']}_{tgttop}_{target}.pkl", "rb"))
    for r in records:
        r["node_id"] = cameras[r["camera_id"]]["node_id"]
        del r["camera_id"]
    records = {i: r for i, r in enumerate(records)}
    labels = pickle.load(open(f"log/{exp_name}/labels/iter_best.pkl", "rb"))
    assert len(records) == len(labels)
    print(target, len(labels))
    return records, labels


class Recover:
    def __init__(self, G: nx.DiGraph, router: Router, shortest_path=False):
        self.G = G
        self.eid2e = {e["id"]: e for _, _, e in G.edges(data=True)}
        self.router = router
        self.shortest_path = shortest_path
        
    def get_cuts(self, rs, adj_range, tm_gap_gate):
        """将1个cluster里的records切分成若干段"""
        points = [(r['node_id'], r['time'], r["id"]) for r in rs]
        points = merge_tm_adj_points(points, adj_range)
        points = merge_tm_adj_points(points, adj_range)
        points.sort(key=lambda x: x[1])
        cuts = cut_distant_points(points, tm_gap_gate)
        return cuts

    def rec_cut(self, cut):
        """
        恢复每段轨迹得到route:
            [(u, ut, rids), [gpss, eids, p], (v, vt, rids), ...]
            摄像头观测点u    路径u->v         摄像头观测点v
            (trick: 用tuple和list区分两种数据类型)
        若u,v相同, gpss取其中1点, eids为[]
        若u,v间不连通, gpss取两端点, eids为[-1]
        """
        route = [cut[0]]
        for (u, ut, _), (v, vt, vi) in zip(cut, cut[1:]):
            if self.shortest_path:
                try:
                    nids = nx.shortest_path(self.G, u, v, weight="length") if u != v \
                        else [u, v]  # 若u,v相同, nids为[u,v]
                    p = 1
                except:
                    nids = [u, v]    # 若u到v间没有路, nids为[u,v]
                    p = 0
            else:
                nids, p = self.router.MAP_routing(u, v, ut, vt, return_route=True)
                nids = [u] + nids + [v]  # 若u,v相同, 或u到v间没有路, nids为[u,v]
            gpss = [(self.G.nodes[nids[0]]["x"], self.G.nodes[nids[0]]["y"])]
            eids = []
            for n1, n2 in zip(nids, nids[1:]):
                if n1 == n2:  # 原地停留, 则gpss取单点, eids为[]
                    assert len(nids) == 2
                else:
                    try:
                        gpss += self.G.edges[n1, n2]["points_gps"][1:]
                        eids.append(self.G.edges[n1, n2]["id"])
                    except:   # 若u,v间不连通, gpss取两端点, eids为[-1]
                        assert len(nids) == 2
                        gpss.append((self.G.nodes[n2]["x"], self.G.nodes[n2]["y"]))
                        eids.append(-1)
            route.append([gpss, eids, p])   # list
            route.append((v, vt, vi))       # tuple
        # 将不连通的route断开
        routes = []
        one_route = [route[0]]
        i = 0
        while i + 2 < len(route):
            p, v = route[i+1], route[i+2]
            if p[1] != [-1]:
                one_route.append(p)
                one_route.append(v)
            else:
                routes.append(one_route)
                one_route = [v] 
            i += 2
        routes.append(one_route)
        return routes
    
    def interpolate_route_by_intersection(self, route):
        """
        在route的每个路口处插值点
        input: [(u, ut, rids), [gpss, eids, p], (v, vt, rids), ...]
        output: [(u, ut, rids), [points, gpss, eids, p], (v, vt, rids), ...]
        其中points为[(nid, t), ...]
        """
        i = 0
        while i + 2 < len(route):
            ut, eids, vt = route[i][1], route[i+1][1], route[i+2][1]
            points = []
            if len(eids) > 1:
                eid2speed = self.router.speed_dicts[int(ut / 3600)]
                ts_exp = [self.eid2e[eid]["length"] / eid2speed[eid] for eid in eids]
                p = (vt - ut) / sum(ts_exp)
                ct = ut
                for eid, t_exp in zip(eids[:-1], ts_exp[:-1]):
                    nid = self.eid2e[eid]["od"][1]
                    ct += t_exp * p
                    points.append((nid, ct))
            route[i+1] = [points] + route[i+1]
            i += 2
        return route

    def rec_cluster(self, rs, adj_range=ADJ_RANGE, tm_gap_gate=TM_GAP_GATE):
        cuts = self.get_cuts(rs, adj_range, tm_gap_gate)
        routes = sum([self.rec_cut(cut) for cut in cuts], [])
        routes = [self.interpolate_route_by_intersection(r) for r in routes]
        return routes


def plot_route(G, route, path):
    m = get_base_map(gaode=True)
    colors = get_colors()
    for i, one_cut_route in enumerate(route):
        color = colors[i % len(colors)]
        for x in one_cut_route:
            if isinstance(x, tuple):
                nid, t, rids = x
                folium.CircleMarker(
                    location=wgs2gcj(G.nodes[nid]["x"], G.nodes[nid]["y"])[::-1],
                    radius=5,
                    color=color,
                    opacity=0.7,
                    fill=True,
                    popup=(time_conventer(round(t)), rids),
                ).add_to(m)
            elif isinstance(x, list):
                points, gpss, eids, p = x
                if len(gpss) > 1:
                    folium.PolyLine(
                        locations=[wgs2gcj(*p)[::-1] for p in gpss],
                        weight=3,
                        color=color,
                        opacity=0.7,
                        popup=round(100*p, 2) 
                    ).add_to(m)
    m.save(path)


def main():
    G = pickle.load(open(f"../data_interface/G_{tgttop}.pkl", 'rb'))
    router = Router(target="all")
    recover = Recover(G=G, router=router)

    # 经过跨区域合并后的全城records及labels
    records = pickle.load(open(f"data/records_{tgttop}_merged.pkl", "rb"))
    labels = pickle.load(open(f"data/labels_{tgttop}_merged.pkl", "rb"))
    print(len(labels))
    cid2rs = defaultdict(list)
    for i, c in enumerate(labels):
        cid2rs[c].append(records[i])
    del records
    del labels

    # 分batch恢复所有类
    cid2rs = [(cid, rs) for cid, rs in cid2rs.items() if len(rs) > 1]
    batch_size = 100000
    n = 0
    for i in tqdm(range(0, len(cid2rs), batch_size)):
        data = cid2rs[i: i+batch_size]
        pickle.dump(data, open(f"data_rec/cid_rs_{tgttop}_{n}.pkl", "wb"))
        n += 1
    del cid2rs

    n = 0
    while os.path.exists(f"data_rec/cid_rs_{tgttop}_{n}.pkl"):
        data = pickle.load(open(f"data_rec/cid_rs_{tgttop}_{n}.pkl", "rb"))
        print(n, len(data))
        results = process_map(
            recover.rec_cluster,
            [x[1] for x in data],
            chunksize=min(100, ceil(len(data) / workers)), 
            max_workers=workers)
        cid2routes = {x[0]: routes for x, routes in zip(data, results)}
        pickle.dump(cid2routes, open(f"data_rec/cid2routes_{tgttop}_{n}.pkl", "wb"))
        del cid2routes
        n += 1


def release_dataset():
    """
    {
        "vid": cid,
        "tid": trip_id,
        "ps": [(node_id, t)],
        "dt": points[0][1], 
        "dr": points[-1][1] - points[0][1],
        "l": length,
    }
    """
    G = pickle.load(open(f"../data_interface/G_{tgttop}.pkl", 'rb'))
    eid2l = {e["id"]: e["length"] for _, _, e in G.edges(data=True)}

    n = 0
    trajs = []
    records = pickle.load(open(f"data/records_{tgttop}_merged.pkl", "rb"))
    records = {r["id"]: r for r in records}
    while os.path.exists(f"data_rec/cid2routes_{tgttop}_{n}.pkl"):
        cid2routes = pickle.load(open(f"data_rec/cid2routes_{tgttop}_{n}.pkl", "rb"))
        print(n, len(cid2routes), sum(len(v) for v in cid2routes.values()))
        for cid, routes in tqdm(cid2routes.items()):
            # [(u, ut, rids), [points, gpss, eids, p], (v, vt, rids), ...]
            routes = [r for r in routes if len(r) > 1]
            trip_id = 0
            for route in routes:
                points = []
                length = 0
                for i in route:
                    if isinstance(i, tuple):
                        points.append(i[:2])
                    else:
                        points += i[0]
                        for eid in i[2]:
                            length += eid2l[eid]
                if length > 0:
                    trajs.append({
                        "vid": cid,
                        "tid": trip_id,
                        "ps": points,
                        "dt": points[0][1],
                        "dr": points[-1][1] - points[0][1],
                        "l": length,
                    })
                    trip_id += 1
        print("trajs:", len({t["vid"] for t in trajs}), len(trajs))
        n += 1
    
    pickle.dump(trajs, open(f"data/trajs_release_{tgttop}.pkl", "wb"))
    

if __name__ == "__main__":
    main()

    release_dataset()
