import argparse
import os
import re
import subprocess
import time

import yaml
from tqdm import tqdm


class Table:
    def __init__(self, m, n):
        self._table = [[""] * n for _ in range(m)]

    @staticmethod
    def count_utf8(s):
        return sum(ord(i) > 128 for i in s)

    def __setitem__(self, pos, value):
        self._table[pos[0]][pos[1]] = str(value)

    def __repr__(self):
        ws = [[len(i) + self.count_utf8(i) for i in j] for j in self._table]
        ws = [max(i) for i in zip(*ws)]
        return "\n".join(
            " ".join(j.ljust(k + 1 - self.count_utf8(j)) for j, k in zip(i, ws))
            for i in self._table
        )

    def write(self, file):
        with open(file, "w") as f:
            f.write("\n".join("\t".join(j for j in i) for i in self._table))


def print_results(results):
    if len(results) <= 1:
        return
    ps = []
    for i in list(results.values())[0]:
        if len({j[i] for j in results.values()}) > 1:
            ps.append(i)
    ps = sorted(ps)
    n = len(ps)
    if n == 0:
        return
    table = Table(1 + n * len(results), 8)
    table[0, 0] = "param"
    table[0, 1] = "type"
    table[0, 2] = "T_noise"
    table[0, 3] = "F_noise"
    table[0, 4] = "P_noise"
    table[0, 5] = "T_recall"
    table[0, 6] = "F_recall"
    table[0, 7] = "P_recall"
    for i, (a, b) in enumerate(results.items()):
        table[1 + i * n, 0] = a
        for j, k in enumerate(ps):
            x, y, z, w = b[k]
            table[1 + i * n + j, 1] = k
            table[1 + i * n + j, 2] = x
            table[1 + i * n + j, 3] = y
            table[1 + i * n + j, 4] = x and f"{x/(x+y)*100:.2f}%"
            table[1 + i * n + j, 5] = z
            table[1 + i * n + j, 6] = w
            table[1 + i * n + j, 7] = z and f"{z/(z+w)*100:.2f}%"
    print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="config file")
    parser.add_argument("-p", type=str, help="parameter:start,end,step")
    parser.add_argument("--load", type=str, help="load previous log")
    parser.add_argument("--cuda", type=str, default="")
    args = parser.parse_args()
    if args.load:
        logs = []
        log = []
        flag_1 = False
        flag_2 = False
        p = None
        trials = []
        for i in open(args.load):
            if flag_2:
                p_, v = i.split(": ")
                assert p is None or p == p_
                p = p_
                trials.append(round(float(v), 7) if "." in v else int(v))
                flag_2 = False
            if i.startswith("---------------"):
                flag_2 = True
            if flag_1 and len(i.strip()) == 0:
                flag_1 = False
                logs.append(log)
                log = []
            if flag_1:
                log.append(i)
            if i.lstrip().startswith("true_noise  fake_noise"):
                flag_1 = True
        assert len(logs) == len(trials)
        results = {}
        for a, log in zip(trials, logs):
            results[a] = {
                i.split()[0]: tuple(int(j) for j in re.split(" +", i)[1:]) for i in log
            }
        print(f"param: {p}")
        print_results(results)
        exit()

    cfg = yaml.load(open(args.c, "rb"), yaml.Loader)
    p, s = args.p.split(":")
    assert p in cfg
    assert cfg["debug_phase"] < 2
    if "." in s:
        a, b, c = [float(i) for i in s.split(",")]
    else:
        s = [int(i) for i in s.split(",")]
        if len(s) == 3:
            a, b, c = s
        else:
            assert len(s) == 2
            a, b = s
            c = 1
    trials = []
    while a <= b:
        trials.append(a)
        a += c
        a = round(a, 7)
    os.makedirs("tmp", exist_ok=True)
    cfg_tmp = f"tmp/{time.time()}.yml"
    results = {}
    with open(time.strftime(f"log/_tuning_%y%m%d_%H%M%S.log"), "a") as out:
        for a in tqdm(trials, dynamic_ncols=True, smoothing=0):
            st = time.time()
            cfg[p] = a
            with open(cfg_tmp, "w") as f:
                yaml.dump(cfg, f, indent=4)
            print("\n" + "-" * 70)
            print(f"{p}: {a}")
            ps = subprocess.Popen(
                f"stdbuf -oL python main.py -c {cfg_tmp} --restore --no-copy --cuda '{args.cuda}'",
                stdout=subprocess.PIPE,
                encoding="utf8",
                shell=True,
            )
            do_print = False
            out.write("\n" + "-" * 70 + "\n")
            out.write(f"{p}: {a}")
            t = []
            for l in ps.stdout:
                out.write(l)
                out.flush()
                if l.startswith("   ") and l.lstrip().startswith("true_noise"):
                    do_print = True
                if do_print:
                    print(l.rstrip())
                    t.append(l.strip())
            results[a] = {
                i.split()[0]: tuple(int(j) for j in re.split(" +", i)[1:])
                for i in t[1:]
            }
            ps.wait()
            print(f"\nTime: {time.time()-st:.3f}s")
            print_results(results)


if __name__ == "__main__":
    main()
