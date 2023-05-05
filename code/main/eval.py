import logging
import pickle
import time
import os
from collections import Counter, defaultdict


def _eval(pair):
    pair.sort(key=lambda x: x[0])
    vid2cid = defaultdict(list)
    gt_size = defaultdict(int)
    cid_size = defaultdict(int)
    for i, j in pair:
        vid2cid[i].append(j)
        cid_size[j] += 1
        if i != -1:
            gt_size[i] += 1
    vid2cid.pop(-1, None)
    assert len(gt_size) == len(vid2cid)
    precision = 0
    recall = 0
    vid_to_cid = {}
    for vid, cids in vid2cid.items():
        cs = [i for i in cids if i != -1]
        if cs:
            cid, cnt = max(Counter(cs).items(), key=lambda x: x[1])
            precision += cnt / cid_size[cid] * gt_size[vid]
            recall += cnt
            vid_to_cid[vid] = cid
    gt_total = sum(gt_size.values())
    precision /= gt_total
    recall /= gt_total
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    expansion = sum(len(set(i)) for i in vid2cid.values()) / len(vid2cid)
    return precision, recall, fscore, expansion, vid_to_cid


def evaluate(records, labels, log=True, save_name=None, folder=None):
    if not isinstance(labels, list):
        labels = labels.tolist()
    if isinstance(records, dict):
        records = [records[i] for i in range(len(records))]
    pair = [
        (-1 if i["vehicle_id"] is None else int(i["vehicle_id"]), j)
        for i, j in zip(records, labels)
    ]
    if save_name is not None:
        path = os.path.dirname(os.path.abspath(__file__)) + "/eval_history"
        if folder is not None:
            path += "/" + folder
            os.makedirs(path, exist_ok=True)
        name = f'{path}/{time.strftime("%Y%m%d_%H%M%S")}_{save_name}'
        pickle.dump(pair, open(name, "wb"))
        logging.info(f"saved to {name}")
    precision, recall, fscore, expansion, vid_to_cid = _eval(pair)
    if log:
        logging.info(
            f"\nclusters: {len(set(labels)-set([-1]))}\norphans: {labels.count(-1)}\nprecision: {precision}\nrecall: {recall}\nfscore: {fscore}\nexpansion: {expansion}"
        )
    return precision, recall, fscore, expansion, vid_to_cid
