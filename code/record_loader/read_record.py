import os
import pickle
import random
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.append("..")
from toputils import *

tgttop, _, config = read_config()
F_DIM = 64 if tgttop == "sz3rd" else 256
PCA_DIM = config["pca"]
NN = 4  # 内存原因, 分批处理
input_path = "../dataset/" + config["dataset"]["record"]
camera_path = f"../data_interface/r2cameras_{tgttop}.pkl"
random.seed(233)

def records_pca(records, k, inplace=True):
    if not inplace:
        records = deepcopy(records)
    car_f = np.array([i[4] for i in records])
    plate_f = np.array([i[5] for i in records if i[5] is not None])
    car_f = PCA(k).fit_transform(car_f)
    if plate_f.size > 0:
        plate_f = PCA(k).fit_transform(plate_f)
    j = 0
    for i, r in enumerate(records):
        r[4] = car_f[i]
        if r[5] is not None:
            r[5] = plate_f[j]
            j += 1
    return records


def rearrange_records(records):
    records_wt_pf = [r for r in records if r[5] is not None]
    print("with plate feature:", len(records_wt_pf))
    records = [r for r in records if r[5] is None]
    print("without plate feature:", len(records))
    random.shuffle(records_wt_pf)
    random.shuffle(records)
    return records_wt_pf + records


def main():
    r2cids = {r: set(c["id"] for c in cams) for r, cams in pickle.load(open(camera_path, "rb")).items()}
    cids_all = set().union(*r2cids.values())
    print("input cameras:", len(cids_all))

    for nn in range(NN):
        print("part:", nn)
        cache_path = f"data/cid2records_{tgttop}_part{nn}.pkl"
        if os.path.exists(cache_path):
            cid2records = pickle.load(open(cache_path, "rb"))
        else:
            print("Loading records...")
            records = np.load(input_path)["arr_0"]
            print("input shape:", records.shape)
            assert records.shape[1] == 2*F_DIM + 4
            N = int(len(records) / NN)
            records = records[nn*N: (nn+1)*N if nn < NN-1 else len(records)]
            cid2records = defaultdict(list)
            for r in tqdm(records): 
                if int(r[2]) in cids_all:
                    rid = int(r[0])
                    vid = None if int(r[1]) == -1 else int(r[1])
                    cid = int(r[2])
                    t = int(r[3])
                    cf = np.asarray(r[4:4+F_DIM])
                    pf = None if np.all(r[4+F_DIM:]==0) else np.asarray(r[4+F_DIM:])
                    assert len(r) == 2*F_DIM + 4
                    cid2records[cid].append([rid, vid, cid, t, cf, pf])
            del records
            pickle.dump(cid2records, open(cache_path, "wb"))

        print("camera with records:", len(cid2records))

        for r, cids in r2cids.items():
            print(r)
            rcds = [rcd for cid in cids for rcd in cid2records.get(cid, [])]
            print("in region records:", len(rcds))
            pickle.dump(rcds, open(f"data/records_{tgttop}_{r}_part{nn}.pkl", "wb"))
        del cid2records
    
    print("Merging part results, PCA, Rearrange...")
    keys = ["id", "vehicle_id", "camera_id", "time", "car_feature", "plate_feature"]
    for r in config["regions"]:
        print(r)
        rcds = []
        for nn in range(NN):
            rcds += pickle.load(open(f"data/records_{tgttop}_{r}_part{nn}.pkl", "rb"))
        if F_DIM > PCA_DIM:
            print("Saving nopca result0...")
            pickle.dump(rcds, open(f"../data_interface/records_nopca_{tgttop}_{r}.pkl", "wb"))
            print("PCA...")
            rcds = records_pca(rcds, PCA_DIM, inplace=True)
            rcds = rearrange_records(rcds)
        rcds = [{k: v for k, v in zip(keys, x)} for x in rcds]
        print("Saving result...")
        pickle.dump(rcds, open(f"../data_interface/records_pca_{config['pca']}_{tgttop}_{r}.pkl", "wb"))

        print("record num:", len(rcds))
        
        gt_cnt = 0
        gt_id = set()
        for x in rcds:
            if x["vehicle_id"]:
                gt_cnt += 1
                gt_id.add(x["vehicle_id"])
        print("gt_cnt:", gt_cnt)
        print("get id cnt:", len(gt_id))


if __name__ == "__main__":
    main()
