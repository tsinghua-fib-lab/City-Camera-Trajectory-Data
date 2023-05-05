import pickle
import sys
from itertools import islice
from math import ceil

from networkx import shortest_simple_paths
from tqdm.contrib.concurrent import process_map

sys.path.append("..")
from toputils import *


K = 10
workers = 32
weight = "length"
global G, Gr


def unit(arg):
    try:
        return arg, list(islice(shortest_simple_paths(Gr, *arg, weight), K))
    except:
        try:
            return arg, list(islice(shortest_simple_paths(G, *arg, weight), K))
        except:
            return None


def main():
    tgttop = read_config()[0]
    G_path = f"../data_interface/G_{tgttop}.pkl"
    r2G_path = f"../data_interface/r2G_{tgttop}.pkl"
    camera_path = f"../data_interface/r2cameras_{tgttop}.pkl"

    global G, Gr
    G = pickle.load(open(G_path, 'rb'))
    r2G = pickle.load(open(r2G_path, 'rb'))
    r2cameras = pickle.load(open(camera_path, "rb"))
    for r, cameras in r2cameras.items():
        print(r)
        Gr = r2G[r]

        camera_nodes = set(x['node_id'] for x in cameras)
        print("camera_nodes:", len(camera_nodes))

        args = [(u, v) for u in camera_nodes for v in camera_nodes if u != v]
        results = process_map(unit, args, chunksize=min(ceil(len(args)/workers), 20), max_workers=workers)
        results = [x for x in results if x]
        shortest_path_results = {uv: paths for uv, paths in results}
        pickle.dump(shortest_path_results, open(f"../data_interface/shortest_path_{tgttop}_{r}.pkl", "wb"))

        # 由于路网连通性的问题, 可能存在部分摄像头之间没有路
        not_path_cnt = 0
        for arg in args:
            if not shortest_path_results.get(arg, None):
                not_path_cnt += 1
        print("no path cnt:", not_path_cnt, "/", len(args))


if __name__ == "__main__":
    main()