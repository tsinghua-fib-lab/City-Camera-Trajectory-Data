"""
合并跨区域的轨迹
"""
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from math import ceil
from math import e as E
from pprint import pprint

import folium
import yaml
from eval import evaluate
from main import cut_distant_points, merge_tm_adj_points
from routing import Router
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils import mean

sys.path.append("..")
from toputils import *

random.seed(233)

tgttop, _, config = read_config()

# t_A = "a24"
# t_1 = "cluster2"
# t_2 = "cluster4"

# t_A = "a25"
# t_1 = "cluster2"
# t_2 = "cluster5"

# t_A = "a15"
# t_1 = "cluster1"
# t_2 = "cluster5"

t_A = "a35"
t_1 = "cluster3"
t_2 = "cluster5"

G = pickle.load(open(f"../data_interface/G_{tgttop}.pkl", "rb"))
r2cams = pickle.load(open(f"../data_interface/r2cameras_{tgttop}.pkl", "rb"))
cams_A = {x["id"]: x for x in r2cams[t_A]}
cams_1 = {x["id"]: x for x in r2cams[t_1]}
cams_2 = {x["id"]: x for x in r2cams[t_2]}

router_A = Router(target=t_A)
router_1 = Router(target=t_1)
router_2 = Router(target=t_2)

ADJ_RANGE = 180
TM_GAP_GATE = 1800
workers = 8
SIM_GATE = 0.8   # 视觉相似度阈值
P_penal = 0.8    # 对不重叠点的惩罚系数
ST_GATE = 0.6    # 时空匹配阈值


def get_most(arr):
    """求数组中出现次数最多的元素"""
    if len(arr) == 1:
        return arr[0]
    # return max(list(Counter(arr).items()), key=lambda x: x[1])[0]
    c = list(Counter(arr).items())
    n = max(x[1] for x in c)
    return random.choice([a for a, b in c if b == n])


def get_records_labels(target, cameras):
    cfg = yaml.load(open(config["regions"][target]["main_config"], "r", encoding="utf8"), Loader=yaml.SafeLoader)
    exp_name = cfg["name"]
    records = pickle.load(open(f"../data_interface/records_pca_{config['pca']}_{tgttop}_{target}.pkl", "rb"))
    for r in records:
        r["node_id"] = cameras[r["camera_id"]]["node_id"]
    if tgttop == "sz2nd":  # 使用PCA前的视觉特征才能跨区域计算相似度
        records_nopca = pickle.load(open(f"../data_interface/records_nopca_{tgttop}_{target}.pkl", "rb"))
        records_nopca = {r[0]: [r[4], r[5]] for r in records_nopca}
        for r in records:
            r["car_feature"] = records_nopca[r["id"]][0]
            r["plate_feature"] = records_nopca[r["id"]][1]
    records = {i: r for i, r in enumerate(records)}
    labels = pickle.load(open(f"log/{exp_name}/labels/iter_best.pkl", "rb"))
    assert len(records) == len(labels)
    print(target, len(labels))
    return records, labels


def find_M(records, labels):
    """
    在辅助区域A内, 对所有聚类结果, 筛选其中跨了两个区域的轨迹M
    计算平均视觉特征
    M进一步划分为从区域1到区域2的M12, 从区域2到区域1的M21
    """
    cid2rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid2rids[c].append(i)

    M12, M21 = [], []
    cnt = 0
    for cid, rids in cid2rids.items():
        if not (len(rids) > 1 and c != -1):
            continue
        points = [(records[i]['node_id'], records[i]['time'], i) for i in rids]
        points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
        points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
        points.sort(key=lambda x: x[1])  # [(cnid, t, idxs)]
        cuts = cut_distant_points(points, tm_gap_gate=TM_GAP_GATE)
        f = None
        for ps in cuts:
            if len(ps) < 2:
                continue
            cnt += 1

            od = None
            a, b = records[ps[0][-1][0]]["camera_id"], records[ps[-1][-1][0]]["camera_id"]
            if a in cams_1 and b in cams_2:
                od = 12
            elif a in cams_2 and b in cams_1:
                od = 21
            if od is None:
                continue

            if f is None:
                fc = [records[rid]["car_feature"] for rid in rids]  # 使用整个类的平均视觉特征
                fc = mean(fc, True)
                fp = [records[rid]["plate_feature"] for rid in rids]
                fp = [x for x in fp if x is not None]
                fp = mean(fp, True) if fp else None
                f = (fc, fp)
            
            ps1, ps2 = [], []  # ps分为在区域1,2中的点
            for p in ps:
                t = records[p[-1][0]]["camera_id"]
                if t in cams_1:
                    ps1.append(p[:2])
                elif t in cams_2:
                    ps2.append(p[:2])
            
            m = {
                "cid": cid,
                "f": f,
                "ps": ps,
                "ps1": ps1,
                "ps2": ps2,
            }
            vids = [records[idx]["vehicle_id"] for p in ps for idx in p[-1]]
            gt_vids = [x for x in vids if x is not None]
            if gt_vids:
                vid, n = max(list(Counter(gt_vids).items()), key= lambda x: x[1])
                if n / len(gt_vids) > 0.5 and n / len(vids) >= 0.5:
                    m["vid"] = vid  # 若是真值车, 记录vid
            (M12 if od == 12 else M21).append(m)

    print(len(M12), "+", len(M21), "/", cnt)
    return M12, M21


def find_HT(records, labels):
    """
    在两侧区域内, 对所有聚类结果, 筛选其中跨了不属于A的区域和属于A的区域的类
    计算平均视觉特征
    分为进入A的和离开A的
    """
    cid2rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid2rids[c].append(i)
    H, T = [], []
    cnt = 0
    for cid, rids in cid2rids.items():
        if not (len(rids) > 1 and c != -1):
            continue
        points = [(records[i]['node_id'], records[i]['time'], i) for i in rids]
        points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
        points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
        points.sort(key=lambda x: x[1])  # [(cnid, t, idxs)]
        cuts = cut_distant_points(points, tm_gap_gate=TM_GAP_GATE)
        f = None
        for ps in cuts:
            if len(ps) < 2:
                continue
            cnt += 1

            ht = None
            a, b = records[ps[0][-1][0]]["camera_id"], records[ps[-1][-1][0]]["camera_id"]
            if a not in cams_A and b in cams_A:
                ht = "h"
                psa = []
                for p in ps[::-1]:
                    if records[p[-1][0]]["camera_id"] in cams_A:
                        psa.append([p[0], p[1]])
                    else:
                        break
            elif a in cams_A and b not in cams_A:
                ht = "t"
                psa = []
                for p in ps:
                    if records[p[-1][0]]["camera_id"] in cams_A:
                        psa.append([p[0], p[1]])
                    else:
                        break
            if ht is None:
                continue

            if f is None:
                fc = [records[rid]["car_feature"] for rid in rids]  # 使用整个类的平均视觉特征
                fc = mean(fc, True)
                fp = [records[rid]["plate_feature"] for rid in rids]
                fp = [x for x in fp if x is not None]
                fp = mean(fp, True) if fp else None
                f = (fc, fp)
            
            r = {
                "cid": cid,
                "f": f,
                "ps": ps,
                "psa": psa  # 首/尾在A中的部分
            }
            vids = [records[idx]["vehicle_id"] for p in ps for idx in p[-1]]
            gt_vids = [x for x in vids if x is not None]
            if gt_vids:
                vid, n = max(list(Counter(gt_vids).items()), key= lambda x: x[1])
                if n / len(gt_vids) > 0.5 and n / len(vids) >= 0.5:
                    r["vid"] = vid  # 若是真值车, 记录vid
            (H if ht == "h" else T).append(r)

    print(len(H), "+", len(T), "/", cnt)
    return H, T


def calc_sim(arg):
    (fc1, fp1), cid2xs = arg
    xcid2sim = {}
    for cid, xs in cid2xs.items():
        fc2, fp2 = xs[0]["f"]
        sim = fc1 @ fc2
        if fp1 is not None and fp2 is not None:
            sim = 0.2 * sim + 0.8 * fp1 @ fp2
        if sim > SIM_GATE:
            xcid2sim[cid] = sim
    return xcid2sim


def calc_all_sim(cid2ms, cid2xs):
    args = [(ms[0]["f"], cid2xs) for ms in cid2ms.values()]
    xcid2sims = process_map(
        calc_sim, args, 
        chunksize=min(1000, ceil(len(args)/workers)), max_workers=workers
    )
    mcid2xcid2sim = {mcid: xcid2sim for mcid, xcid2sim in zip(cid2ms, xcid2sims)}
    return mcid2xcid2sim


def pre_match(H, M, T):
    def get_cid2xs(X):
        cid2xs = defaultdict(list)
        for x in X:
            cid2xs[x["cid"]].append(x)
        return cid2xs
    cid2hs = get_cid2xs(H)
    cid2ms = get_cid2xs(M)
    cid2ts = get_cid2xs(T)
    cids2sim_mh = calc_all_sim(cid2ms, cid2hs)
    cids2sim_mt = calc_all_sim(cid2ms, cid2ts)
    return cid2hs, cid2ms, cid2ts, cids2sim_mh, cids2sim_mt


def match_HMT(cid2hs, cid2ms, cid2ts, cids2sim_mh, cids2sim_mt, od):
    """
    匹配h-m-t
    视觉特征
    时空重合点: 连续
    间断点: 概率
    """
    def get_score(a, b):
        """score > 1/2 when a > 0"""
        return (1 - (1 / (E ** a + 1))) * P_penal ** b

    def match_hm(h, m):
        psm = m["ps1"] if od == 12 else m["ps2"]
        psh = h["psa"]
        j_matched = set()
        for nid1, t1 in psm:
            matched = []
            for j, p2 in enumerate(psh):
                if j in j_matched:
                    continue
                nid2, t2 = p2
                if nid1 == nid2:
                    tgap = abs(t1 - t2)
                    if tgap < ADJ_RANGE:
                        matched.append([tgap, j])
            if matched:
                j_matched.add(min(matched, key=lambda x: x[0])[1])
        a = len(j_matched)               # 区域A中, 两轨迹的重合点数
        b = len(psm) + len(psh) - 2 * a  # 区域A中, 两轨迹的非重合点数
        if a > 0:
            return get_score(a, b)
        else:
            u, ut = psh[-1]
            v, vt = psm[0]
            if ut >= vt:
                return 0
            return router_A.MAP_routing(u, v, ut, vt) * P_penal ** b

    def match_mt(m, t):
        psm = m["ps2"] if od == 12 else m["ps1"]
        pst = t["psa"]
        j_matched = set()
        for nid1, t1 in psm:
            matched = []
            for j, p2 in enumerate(pst):
                if j in j_matched:
                    continue
                nid2, t2 = p2
                if nid1 == nid2:
                    tgap = abs(t1 - t2)
                    if tgap < ADJ_RANGE:
                        matched.append([tgap, j])
            if matched:
                j_matched.add(min(matched, key=lambda x: x[0])[1])
        a = len(j_matched)               # 区域A中, 两轨迹的重合点数
        b = len(psm) + len(pst) - 2 * a  # 区域A中, 两轨迹的非重合点数
        if a > 0:
            return get_score(a, b)
        else:
            u, ut = psm[-1]
            v, vt = pst[0]
            if ut >= vt:
                return 0
            return router_A.MAP_routing(u, v, ut, vt) * P_penal ** b

    ch_matched = set()
    ct_matched = set()
    results_ch_cm_ct = []
    for cm, ms in tqdm(cid2ms.items()):
        chs_candidates = []
        for ch, vi_score in cids2sim_mh[cm].items():
            if ch in ch_matched:
                continue
            hs = cid2hs[ch]
            st_scores = [
                (
                    match_hm(h, m), m["ps"], h["ps"], 
                    m.get("vid", None), h.get("vid", None)
                ) 
                for m in ms for h in hs
            ]
            st_score, mps, hps, mvid, hvid = max(st_scores, key=lambda x: x[0])
            if st_score > ST_GATE:
                # print(st_score)
                chs_candidates.append([vi_score * st_score, ch, mps, hps, mvid, hvid])
        # print(len(cids2sim_mh[cm]), len(chs_candidates))
        if chs_candidates:
            sch, ch, mps_h, hps, mvid, hvid = max(chs_candidates, key=lambda x: x[0])
            ch_matched.add(ch)
        else:
            sch, ch, mps_h, hps, mvid, hvid = None, None, None, None, None, None

        cts_candidates = []
        for ct, vi_score in cids2sim_mt[cm].items():
            if ct in ct_matched:
                continue
            ts = cid2ts[ct]
            st_scores = [
                (
                    match_mt(m, t), m["ps"], t["ps"],
                    m.get("vid", None), t.get("vid", None)
                )
                for m in ms for t in ts
            ]
            st_score, mps, tps, mvid, tvid = max(st_scores, key=lambda x: x[0])
            if st_score > ST_GATE:
                # print(st_score)
                cts_candidates.append([vi_score * st_score, ct, mps, tps, mvid, tvid])
        # print(len(cids2sim_mt[cm]), len(cts_candidates))
        if cts_candidates:
            sct, ct, mps_t, tps, mvid, tvid = max(cts_candidates, key=lambda x: x[0])
            ct_matched.add(ct)
        else:
            sct, ct, mps_t, tps, mvid, tvid = None, None, None, None, None, None
        
        if not (ch is None and ct is None):
            results_ch_cm_ct.append({
                "cids": [ch, cm, ct],
                "pss": [hps, mps_h, mps_t, tps],
                "scores": [sch, sct],
                "vids": [hvid, mvid, tvid]
            })

    print("matched h:", len([x for x in results_ch_cm_ct if x["cids"][0] is not None]), "/", len(cid2hs))
    print("matched t:", len([x for x in results_ch_cm_ct if x["cids"][-1] is not None]), "/", len(cid2ts))
    print("matched m:", len(results_ch_cm_ct), "/", len(cid2ms))
    return results_ch_cm_ct


def recover_route(ps, router):
    route = [ps[0]]  # [(cnid, t, rids), [gpss, p], (cnid, t, rids), ...]
    for (u, ut, _), (v, vt, vi) in zip(ps, ps[1:]):
        nids, p = router.MAP_routing(u, v, ut, vt, return_route=True)
        nids = [u] + nids + [v]
        gpss = [(G.nodes[nids[0]]["x"], G.nodes[nids[0]]["y"])]
        for n1, n2 in zip(nids, nids[1:]):
            if n1 == n2:
                assert len(nids) == 2
                pass
            else:
                try:
                    gpss += G.edges[n1, n2, 0]["points_gps"][1:]
                except:
                    gpss.append((G.nodes[n2]["x"], G.nodes[n2]["y"]))
        route.append([gpss, p])   # list
        route.append((v, vt, vi)) # tuple
    return route


def plot(one_result, od, gt_trajs):
    hps, mps_h, mps_t, tps = one_result["pss"]
    sch, sct = one_result["scores"]
    hvid, mvid, tvid = one_result["vids"]
    assert hvid == mvid == tvid
    print("vehile:", mvid, "h,m_h,m_t,t ps:", len(hps), len(mps_h), len(mps_t), len(tps))

    same_m = mps_h == mps_t

    if od == 12:
        router_h, router_t = router_1, router_2
    else:
        router_h, router_t = router_2, router_1
    rh = recover_route(hps, router_h)
    rm_h = recover_route(mps_h, router_A)
    rm_t = recover_route(mps_t, router_A)
    rt = recover_route(tps, router_t)
    
    m = folium.Map(location=[22.5643, 113.9953], control_scale=True)
    m.add_child(folium.LatLngPopup())
    regions = config["regions"]
    for rn, r in regions.items():
        folium.PolyLine(
            locations=[p[::-1] for p in json.loads(r["bound"])],
            popup=rn,
            color="green" if "a" in rn else "red" 
        ).add_to(m)
    for traj in gt_trajs:
        locs = [p[:2][::-1] for i, p in enumerate(traj) if i % 3 == 0]
        for loc in locs:
            folium.CircleMarker(
                location=loc,
                radius=1,
                color = "black",
                opacity=0.5
            ).add_to(m)

    if same_m:
        rs, cs = [rh, rm_h, rt], ["red", "green", "blue"]
    else:
        rs, cs = [rh, rm_h, rm_t, rt], ["red", "green", "green", "blue"]
    for r, c in zip(rs, cs):
        for x in r:
            if isinstance(x, tuple):
                nid, t = x[:2]
                folium.CircleMarker(
                    location=[G.nodes[nid]["y"], G.nodes[nid]["x"]],
                    radius=5,
                    color=c,
                    opacity=0.5,
                    fill=True,
                    popup=time_conventer(round(t)),
                ).add_to(m)
            elif isinstance(x, list):
                gpss, p = x
                if len(gpss) > 1:
                    folium.PolyLine(
                        locations=[gps[::-1] for gps in gpss],
                        weight=3,
                        color=c,
                        opacity=0.5,
                        popup=round(100*p, 2) 
                    ).add_to(m)

    m.save(f"figure/traj_merge/traj_merge_{tgttop}_{mvid}_{round(100*sch)}_{round(100*sct)}.html")


def main():
    # 找A中的M
    # 找1, 2中的H, T
    path = f"data/MHT_{tgttop}_{t_A}.pkl"
    if os.path.exists(path):
        M12, M21, H1, T1, H2, T2 = pickle.load(open(path, "rb"))
    else:
        records_A, labels_A = get_records_labels(t_A, cams_A)
        records_1, labels_1 = get_records_labels(t_1, cams_1)
        records_2, labels_2 = get_records_labels(t_2, cams_2)
        M12, M21 = find_M(records_A, labels_A)
        H1, T1 = find_HT(records_1, labels_1)
        H2, T2 = find_HT(records_2, labels_2)
        pickle.dump((M12, M21, H1, T1, H2, T2), open(path, "wb"))
    for x in [M12, M21, H1, T1, H2, T2]:
        print(len(x))

    # 预计算视觉相似度
    path = f"data/MHT_pre_match_{tgttop}_{t_A}.pkl"
    if os.path.exists(path):
        pre12, pre21 = pickle.load(open(path, "rb"))
    else:
        pre12 = pre_match(H1, M12, T2)
        pre21 = pre_match(H2, M21, T1)
        pickle.dump((pre12, pre21), open(path, "wb"))

    # 时空匹配
    cid2hs, cid2ms, cid2ts, cids2sim_mh, cids2sim_mt = pre12
    merged_1A2 = match_HMT(cid2hs, cid2ms, cid2ts, cids2sim_mh, cids2sim_mt, od=12)  # 217, 117, 109,  4207
    cid2hs, cid2ms, cid2ts, cids2sim_mh, cids2sim_mt = pre21
    merged_2A1 = match_HMT(cid2hs, cid2ms, cid2ts, cids2sim_mh, cids2sim_mt, od=21)  # 242, 105, 144,  4135
    pickle.dump((merged_1A2, merged_2A1), open(f"data/traj_merge_result_{tgttop}_{t_A}.pkl", "wb"))


def plot_case():
    vid2trajs = pickle.load(open("../data_interface/gt_vid2trajs_3rd.pkl", "rb"))
    merged_1A2, merged_2A1 = pickle.load(open(f"data/traj_merge_result_{tgttop}_{t_A}.pkl", "rb"))
    for x in merged_1A2:
        hvid, mvid, tvid = x["vids"]
        if mvid is not None and hvid == mvid == tvid:
            plot(x, od=12, gt_trajs=vid2trajs[mvid])
    for x in merged_2A1:
        hvid, mvid, tvid = x["vids"]
        if mvid is not None and hvid == mvid == tvid:
            plot(x, od=21, gt_trajs=vid2trajs[mvid])


def eval_traj_merge():
    """评估轨迹合并"""
    def eval_merge(H, M, T, merged):
        h_vids = {x["vid"] for x in H if "vid" in x}
        m_vids = {x["vid"] for x in M if "vid" in x}
        t_vids = {x["vid"] for x in T if "vid" in x}
        gt_mh = m_vids & h_vids
        gt_mt = m_vids & t_vids

        mh, mt = set(), set()
        mh_false, mt_false = set(), set()
        for x in merged:
            hvid, mvid, tvid = x["vids"]
            if mvid is not None:
                if mvid == hvid:
                    mh.add(mvid)
                else:
                    mh_false.add(mvid)
                if mvid == tvid:
                    mt.add(mvid)
                else:
                    mt_false.add(mvid)
        # mh_false -= mh
        # mt_false -= mt

        # recall的分子, 分母
        hra, hrb = len(mh & gt_mh), len(gt_mh)
        tra, trb = len(mt & gt_mt), len(gt_mt)
        ra, rb = hra + tra, hrb + trb
        # precision的分子, 分母
        hpa, hpb = len(mh & gt_mh), len(mh) + len(mh_false)
        tpa, tpb = len(mt & gt_mt), len(mt) + len(mt_false)
        pa, pb = hpa + tpa, hpb + tpb

        # print("recall:", ra / rb)
        # print("precision:", pa / pb)
        return ra, rb, pa, pb

    merged_1A2, merged_2A1 = pickle.load(open(f"data/traj_merge_result_{tgttop}_{t_A}.pkl", "rb"))
    M12, M21, H1, T1, H2, T2 = pickle.load(open(f"data/MHT_{tgttop}_{t_A}.pkl", "rb"))
    ra1, rb1, pa1, pb1 = eval_merge(H1, M12, T2, merged_1A2)
    ra2, rb2, pa2, pb2 = eval_merge(H2, M21, T1, merged_2A1)
    print("recall:", (ra1 + ra2) / (rb1 + rb2))
    print("precision:", (pa1 + pa2) / (pb1 + pb2))


def eval_cluster_merge():
    """评估轨迹合并带来的聚类合并"""
    t_pairs = [
        ("a24", "cluster2", "cluster4"),
        ("a25", "cluster2", "cluster5"),
        ("a15", "cluster1", "cluster5"),
        ("a35", "cluster3", "cluster5"),
    ]
    cids_groups = []  # 每个group是: {(t1, cid1), (t2, cid2), ...}
    t2records = {}
    t2labels = {}
    for t_A, t_1, t_2 in t_pairs:
        merged_1A2, merged_2A1 = pickle.load(open(f"data/traj_merge_result_{tgttop}_{t_A}.pkl", "rb"))
        print("merged hmt num:", len(merged_1A2) + len(merged_2A1))
        print(t_A)
        records_A, _ = get_records_labels(t_A, {x["id"]: x for x in r2cams[t_A]})
        records_1, labels_1 = get_records_labels(t_1, {x["id"]: x for x in r2cams[t_1]})
        records_2, labels_2 = get_records_labels(t_2, {x["id"]: x for x in r2cams[t_2]})
        records_A = [records_A[i] for i in range(len(records_A))]
        records_1 = [records_1[i] for i in range(len(records_1))]
        records_2 = [records_2[i] for i in range(len(records_2))]
        t2records[t_1], t2labels[t_1] = records_1, labels_1
        t2records[t_2], t2labels[t_2] = records_2, labels_2
        rid2cid_1 = {r["id"]: cid for r, cid in zip(records_1, labels_1)}  # id, 不是idx, 用于找到M的首尾点在区域1,2中的cid
        rid2cid_2 = {r["id"]: cid for r, cid in zip(records_2, labels_2)}

        cid1_to_cid2 = defaultdict(list)      # h-m-t
        cid1_to_cid2_weak = defaultdict(list) # h-m或m-t
        for merged, od in zip([merged_1A2, merged_2A1], [12, 21]):
            for x in merged:
                # "cids": [ch, cm, ct],
                # "pss": [hps, mps_h, mps_t, tps],
                # "scores": [sch, sct],
                # "vids": [hvid, mvid, tvid]
                ch, _, ct = x["cids"]
                if ch is None and ct is None:
                    continue
                weak = False
                if ch is not None and ct is None:    # m的最后1点所在T区域中的cid作为ct
                    weak = True
                    mp = x["pss"][1][-1]
                    rid2cid = rid2cid_2 if od == 12 else rid2cid_1
                    ct = get_most([rid2cid[records_A[idx]["id"]] for idx in mp[-1]])
                elif ch is None and ct is not None:  # m的第1点所在H区域中的cid作为ch
                    weak = True
                    mp = x["pss"][2][0]
                    rid2cid = rid2cid_1 if od == 12 else rid2cid_2
                    ch = get_most([rid2cid[records_A[idx]["id"]] for idx in mp[-1]])
                c1, c2 = (ch, ct) if od == 12 else (ct, ch)
                (cid1_to_cid2_weak if weak else cid1_to_cid2)[c1].append(c2)
        # 得到一组t_pairs下, 区域1和2的cid的合并结果c1_to_c2
        c1_to_c2 = {c1: get_most(c2s) for c1, c2s in cid1_to_cid2.items()}
        for c1, c2s in cid1_to_cid2_weak.items():
            if c1 not in c1_to_c2:
                c1_to_c2[c1] = get_most(c2s)
        print("merged c1-c2 num:", len(c1_to_c2))
        # 添加到所有t_pairs的最终结果cids_groups里
        for c1, c2 in tqdm(c1_to_c2.items()):
            tc1, tc2 = (t_1, c1), (t_2, c2)
            for group in cids_groups:
                if tc1 in group:
                    group.add(tc2)
                    break
                elif tc2 in group:
                    group.add(tc1)
                    break
            else:
                cids_groups.append({tc1, tc2})
        print("merged clusteds:", sum(len(g) for g in cids_groups))

    # 综合所有区域一起算聚类指标
    offset = 0
    t2offset = {}
    for t, labels in t2labels.items():  # 对label做offset
        assert all(l >= 0 for l in labels)
        if offset == 0:
            t2offset[t] = 0
            offset = max(labels) + 1
            continue
        offset_add = max(labels) + 1
        t2labels[t] = [l + offset for l in labels]
        t2offset[t] = offset
        offset += offset_add

    for t, labels in t2labels.items():
        records = t2records[t]
        evaluate(records,labels)

    labels = sum(t2labels.values(), [])
    print(len(labels))
    records = sum([t2records[t] for t in t2labels] ,[])
    assert len(records) == len(labels)
    print("cluster num before merge:", len(set(labels)))
    print("metrics before merge:")
    p, r, f, e, _ = evaluate(records,labels)
    json.dump([p, r, f, e], open(f"log/metrics_{tgttop}_no_merge.json", "w"))

    label_mapping = {}
    new_label = max(labels) + 1
    for group in cids_groups:
        for t, l in group:
            label_mapping[l + t2offset[t]] = new_label
        new_label += 1
    print("labels changed:", len(label_mapping))
    labels = [label_mapping.get(l, l) for l in labels]
    print("cluster num merged:", len(set(labels)))
    p, r, f, e, _ = evaluate(records,labels)
    json.dump([p, r, f, e], open(f"log/metrics_{tgttop}_merged.json", "w"))

    # 保存合并后的全城records和labels, 供最终轨迹恢复使用
    keys = {"node_id", "time", "id", "vehicle_id"}
    records = [{k: v for k, v in r.items() if k in keys} for r in records]
    pickle.dump(records, open(f"data/records_{tgttop}_merged.pkl", "wb"))
    pickle.dump(labels, open(f"data/labels_{tgttop}_merged.pkl", "wb"))


if __name__ == "__main__":
    main()

    # plot_case()

    # eval_traj_merge()

    eval_cluster_merge()