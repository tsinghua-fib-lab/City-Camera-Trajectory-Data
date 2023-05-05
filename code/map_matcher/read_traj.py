import json
import pickle
import random
import sys
from collections import defaultdict

import folium
from eviltransform import distance as gps_distance
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

sys.path.append("..")
from toputils import *

random.seed(233)

# 输出轨迹为: [[lon, lat, t], ...]

def in_bound(lon, lat):
    global bounds
    for min_lon, max_lon, min_lat, max_lat in bounds:
        if min_lon < lon < max_lon and min_lat < lat < max_lat:
            return True
    return False


def in_which_poly(xy):
    global polys
    p = Point(xy)
    for i, poly in enumerate(polys):
        if p.covered_by(poly):
            return i
    return -1


def _process(batch):
    trajs = []
    for item in batch:
        traj = []
        for t, lon, lat in item:
            # 如果超出范围就丢掉
            # 如果隔太远就截断
            if in_bound(lon, lat):
                if traj and gps_distance(traj[-1][1], traj[-1][0], lat, lon) > 200:
                    trajs.append(traj)
                    traj = [[lon, lat, t]]
                else:
                    traj.append([lon, lat, t])
            else:
                if traj:
                    trajs.append(traj)
                    traj = []
        if traj:
            trajs.append(traj)

    trajs_ = []
    for traj in trajs:
        if len(traj) > 3:
            t_ = traj[:1]
            # 去除太密集的点，5s+30m之内
            for a, b in zip(traj[1:], traj[2:]):
                if (
                    a[2] - t_[-1][2] < 5
                    and gps_distance(*a[:2][::-1], *t_[-1][:2][::-1]) < 30
                    and b[2] - t_[-1][2] < 30
                    and gps_distance(*b[:2][::-1], *t_[-1][:2][::-1]) < 100
                ):
                    continue
                t_.append(a)
            t_.append(b)
            if len(t_) > 3 and t_[-1][-1] - t_[0][-1] > 60:
                trajs_.append(t_)

    trajs_new = []
    for traj in trajs_:
        traj_new = [traj[0]]
        for p in traj[1:]:
            if p[-1] - traj_new[-1][-1] > 5:
                traj_new.append(p)
        if len(traj_new) > 10 and traj[-1][-1] - traj[0][-1] > 180:
            trajs_new.append(traj_new)

    return trajs_new


def read(input_path, workers):
    """
    读取并处理高德轨迹
    筛选bbox内轨迹
    由于轨迹本身跳变或跨越bbox导致的跳变, 切分成多段
    进一步去除时空上太密集的点
    """
    global bounds
    data = pickle.load(open(input_path, "rb"))  # [[t, lon, lat], ...]
    print("orig traj:", len(data))
    batch_size = 1000
    if workers == 1:
        trajs = []
        for i in tqdm(range(0, len(data), batch_size)):
            trajs += _process(data[i : i + batch_size])
    elif workers > 1:
        trajs = [
            j
            for i in process_map(
                _process,
                [data[i : i + batch_size] for i in range(0, len(data), batch_size)],
                chunksize=1,
                dynamic_ncols=True,
                max_workers=workers,
            )
            for j in i
        ]
    return trajs


def plot(m, trajs, path):
    folium.LatLngPopup().add_to(m)
    for traj in trajs:
        folium.PolyLine(
            locations=[[lat, lon] for lon, lat, t in traj],
        ).add_to(m)
        for lon, lat, t in traj:
            folium.CircleMarker(
                location=[lat, lon],
                radius=3
            ).add_to(m)
    m.save(path)


def read_gt(vid2trajs):
    """
    读深圳第三次轨迹真值, 准备路网匹配
    """
    vid2trajs_new = {}
    for vid, trajs in vid2trajs.items():
        trajs_cut = []  # 如果超出范围就丢掉, 如果隔太远就截断
        for item in trajs:
            traj = []
            for lon, lat, t in item:
                if in_bound(lon, lat):
                    if traj and gps_distance(traj[-1][1], traj[-1][0], lat, lon) > 200:
                        trajs_cut.append(traj)
                        traj = [[lon, lat, t]]
                    else:
                        traj.append([lon, lat, t])
                else:
                    if traj:
                        trajs_cut.append(traj)
                        traj = []
            if traj:
                trajs_cut.append(traj)

        trajs_new = []
        for traj in trajs_cut:
            traj_new = [traj[0]]
            for p in traj[1:-1]:
                if p[-1] - traj_new[-1][-1] > 5:  # 采样率不用太高
                    traj_new.append(p)
            traj_new.append(traj[-1])
            if len(traj_new) > 3 and traj[-1][-1] - traj[0][-1] > 30:
                trajs_new.append(traj_new)
        if trajs_new:
            vid2trajs_new[vid] = trajs_new
    print(len(vid2trajs_new))

    return vid2trajs_new


def main():
    tgttop, _, config, projector = read_config(proj=True)

    global bounds, polys
    r_bound_poly_areas = []
    for r, v in config["regions"].items():
        ps = json.loads(v["bound"])
        lons, lats = zip(*ps)
        min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)
        min_x, min_y = projector(min_lon, min_lat)
        min_lon, min_lat = projector(min_x - 100, min_y - 100, inverse=True)
        max_x, max_y = projector(max_lon, max_lat)
        max_lon, max_lat = projector(max_x + 100, max_y + 100, inverse=True)
        bound = [min_lon, max_lon, min_lat, max_lat]
        poly = Polygon([projector(*p) for p in ps])
        r_bound_poly_areas.append([r, bound, poly, poly.area])
    r_bound_poly_areas.sort(key=lambda x: -x[-1])  # 优先匹配面积大的区域
    regions, bounds, polys, _ = zip(*r_bound_poly_areas)

    # # 读真值GPS轨迹
    # vid2trajs = pickle.load(open(f"../data_interface/gt_vid2trajs_{tgttop}.pkl", "rb"))
    # vid2trajs = read_gt(vid2trajs)
    # pickle.dump(vid2trajs, open(f"data/gt_vid2trajs_{tgttop}.pkl", "wb"))
    # # vid_trajs = [(vid, traj) for vid, trajs in vid2trajs.items() for traj in trajs]
    # # vids, trajs = zip(*vid_trajs)

    # 读高德GPS轨迹
    input_paths = [f"orig_data/traj_gaode_sz0824_part{n}.pkl" for n in range(4)]
    for i, input_path in enumerate(input_paths):
        print(input_path)
        output_path = f"data/trajs_sz_filtered_part{i}.pkl"
        trajs = read(input_path, workers=1)  # 内存原因不宜多线程
        print("filtered traj:", len(trajs))
        pickle.dump(trajs, open(output_path, "wb"))

        # m = get_base_map()
        # plot(m, random.sample(trajs, 10) if len(trajs) > 10 else trajs, f"figure/trajs_gps_{tgttop}.html")
        # # plot(m, random.sample(trajs, 10) if len(trajs) > 10 else trajs, f"figure/gt_trajs_gps_{target}.html")

    # 按区域分组, 则路网匹配时直接往小路网上匹配, 加快匹配速度
    r2trajs = defaultdict(list)
    output_paths = [f"data/trajs_sz_filtered_part{n}.pkl" for n in range(4)]
    for output_path in output_paths:
        trajs = pickle.load(open(output_path, "rb"))
        for t in tqdm(trajs):
            i = in_which_poly(projector(*t[int(len(t)/2)][:2]))
            if i == -1:
                r2trajs["all"].append(t)  # 不在区域内的轨迹, 可能位于区域边界处, 到时候往大路网上匹配
            else:
                r2trajs[regions[i]].append(t)
        del trajs
    print("traj num for each region:")
    pprint({r: len(trajs) for r, trajs in r2trajs.items()})
    for r, trajs in r2trajs.items():
        pickle.dump(trajs, open(f"data/trajs_{tgttop}_{r}.pkl", "wb"))


if __name__ == "__main__":
    main()
