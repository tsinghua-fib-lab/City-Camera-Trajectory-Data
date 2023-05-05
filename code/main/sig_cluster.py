"""
topK预搜索 + 逐步分类聚类算法
"""
import argparse
import logging
import pickle
import random
import sys
from collections import defaultdict
from functools import partial

import coloredlogs
import faiss
import numpy as np
from torch.cuda import device_count
from tqdm import tqdm

tqdm = partial(tqdm, dynamic_ncols=True)

from eval import evaluate

coloredlogs.install(fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s")
print(f"Available GPUs: {device_count()}")


def normalize(feature):
    return feature / (np.linalg.norm(feature) + 1e-12)


class FlatSearcher:
    def __init__(self, gpus=(0,1,2), feat_len=256):
        dc = device_count()
        while not all(i < dc for i in gpus):
            print(f"\nError: insufficient gpu, {gpus}/{dc}", file=sys.stderr)
            gpus = list(map(int, input("Please re-enter:").split(",")))
        gpus = [i for i in gpus if i < dc]
        if gpus:
            flat_config = []
            for i in gpus:
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = i
                flat_config.append(cfg)
            res = [faiss.StandardGpuResources() for _ in range(len(gpus))]
            indexes = [
                faiss.GpuIndexFlatIP(a, feat_len, b) for a, b in zip(res, flat_config)
            ]
            self.index = faiss.IndexProxy()
            for sub_index in indexes:
                self.index.addIndex(sub_index)
        else:
            self.index = faiss.IndexFlatL2(feat_len)

    def search_by_topk(self, query, gallery, topk=16):
        self.index.reset()
        self.index.add(gallery)
        topk_scores, topk_idxs = self.index.search(query, topk)
        assert np.all(topk_idxs >= 0)
        return topk_scores, topk_idxs


class SigCluster:
    def __init__(self, feature_dims=[256, 256], gpus=(0,1,2)):
        self.searchers = {i: FlatSearcher(gpus, i) for i in set(feature_dims)}
        self.f_dims = feature_dims

    def fit(
        self,
        data,
        initial_labels=None,
        weights=[0.1, 0.9],
        similarity_threshold=0.88,
        topK=128,
        zipped_data=True,
        f_topks_in=None,
        f_topks_out=None,
        normalized=False,
    ):
        """
        data=[[f1, f2, ...], ...]，如果zipped_data=False，那么data=[[f1, ...], [f2, ...], ...]
        weights和topK可以是列表；也可以是一个数，表示对所有的特征都取这个值

        在f_topks_in中传入list，对于非None项会直接用作f_topk而不再搜索
        在f_topks_out中传入一个list，用于获取可选的f_topks返回值
        """
        if isinstance(weights, float) or isinstance(weights, int):
            weights = [weights] * len(self.f_dims)
        else:
            assert len(weights) == len(self.f_dims)
        if isinstance(topK, int):
            topK = [topK] * len(self.f_dims)
        else:
            assert len(topK) == len(self.f_dims)
        if f_topks_in is None:
            f_topks_in = [None] * len(topK)
        else:
            assert len(f_topks_in) == len(topK)

        logging.info("Normalize")
        if zipped_data:
            N = len(data)
            data = [
                [None if j is None else normalize(j) for j in i] for i in tqdm(data)
            ]
            data_ = list(zip(*data))
        else:
            N = len(data[0])
            if normalized:
                for j in data[-1]:
                    l = np.linalg.norm(j)
                    assert np.allclose(l, 1), (l, j)
            data_ = [
                [None if j is None else normalize(j) for j in tqdm(i)] for i in data
            ]
            data = zip(*data_)
        # 聚合非None特征用于搜索
        logging.info("Gather features")
        fs = []
        f_ids = []
        for i in data_:
            f_id, f = zip(*((j, k) for j, k in enumerate(i) if k is not None))
            fs.append(np.array(f))
            f_ids.append(f_id)

        # 搜索每个特征的topk
        logging.info("Search topk")
        f_topks = [
            f_in
            or [
                [f_id[k] for k in j if k < i]
                for i, j in enumerate(
                    self.searchers[dim].search_by_topk(f, f, topk)[1].tolist()
                )
            ]
            for f_in, f, f_id, dim, topk in zip(
                f_topks_in, fs, f_ids, self.f_dims, tqdm(topK)
            )
        ]
        assert all(len(i[0]) == 0 for i in f_topks)
        if f_topks_out is not None:
            f_topks_out.append(f_topks)

        # 计算每个record的topk列表
        logging.info("Flatten topk")
        topks = [[] for _ in range(N)]
        for f_topk, f_id in zip(f_topks, f_ids):
            for i, topk in zip(f_id, f_topk):
                topks[i] += topk

        # 聚类
        logging.info("Clustering")
        cf_means = {}
        cfs = {}
        if initial_labels is None:
            cids = [-1] * N
        else:
            cids = initial_labels
            cid2records = defaultdict(list)
            for cid, record in zip(cids, data):
                if cid >= 0:
                    cid2records[cid].append(record)
            for cid, rs in cid2records.items():
                tmp = cfs[cid] = [[j for j in i if j is not None] for i in zip(*rs)]
                cf_means[cid] = [
                    normalize(np.mean(t, axis=0)) if len(t) else None for t in tmp
                ]
        for i, (record, topk) in enumerate(zip(data, tqdm(topks))):
            if cids[i] >= 0:
                continue
            cs = {cids[i] for i in topk}
            best_cid = -1
            best_sim = -1
            for c in cs:
                w_total = 0
                sim = 0
                for w, a, b in zip(weights, record, cf_means[c]):
                    if a is not None and b is not None:
                        sim += a @ b * w
                        w_total += w
                sim /= w_total
                if sim > best_sim:
                    best_sim = sim
                    best_cid = c
            if best_cid >= 0 and best_sim >= similarity_threshold:
                cids[i] = best_cid
                cf = cfs[best_cid]
                cf_mean = cf_means[best_cid]
                for j, k in enumerate(record):
                    if k is not None:
                        if cf[j]:
                            cf[j].append(k)
                            cf_mean[j] = normalize(np.mean(cf[j], axis=0))
                        else:
                            cf[j] = [k]
                            cf_mean[j] = k
            else:
                cid = len(cf_means)
                cids[i] = cid
                cf_means[cid] = list(record)
                cfs[cid] = [[] if j is None else [] for j in record]
        logging.info("done")
        return cids
