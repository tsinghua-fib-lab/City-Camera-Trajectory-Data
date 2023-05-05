import traceback
import threading
import time
from tqdm import tqdm
import random
import multiprocessing as mp


def thread_map(
    f,
    iter_,
    num_workers=4,
    unpack=False,
    use_tqdm=True,
    tqdm_leave=True,
    shuffle=False,
    disable=False,
):
    if disable:
        it = tqdm(iter_, dynamic_ncols=True, leave=tqdm_leave, disable=not use_tqdm)
        if unpack:
            return [f(*i) for i in it]
        return list(map(f, it))
    ret = []
    alive = []
    err = []

    def _job(_index, _args):
        try:
            if unpack:
                ret.append([_index, f(*_args)])
            else:
                ret.append([_index, f(_args)])
        except:
            err.append(traceback.format_exc())

    iter_ = list(enumerate(iter_))
    if shuffle:
        random.shuffle(iter_)
    with tqdm(
        total=len(iter_),
        dynamic_ncols=True,
        leave=tqdm_leave,
        disable=not use_tqdm,
        miniters=0,
    ) as bar:
        try:
            for i in iter_:
                t = threading.Thread(target=_job, args=i)
                t.start()
                alive.append(t)
                time.sleep(0.01)
                c = len(alive)
                while len(alive) >= num_workers:
                    alive = [i for i in alive if i.is_alive() or i.join()]
                    bar.update(0)
                    assert not err
                    time.sleep(0.1)
                bar.update(c - len(alive))
            while alive:
                c = len(alive)
                alive = [i for i in alive if i.is_alive() or i.join()]
                bar.update(c - len(alive))
                assert not err
                time.sleep(0.1)
        except AssertionError:
            print(err[0])
            exit()

    ret.sort(key=lambda x: x[0])
    return [i for _, i in ret]


def _job_mp(_index, _ret, _err, _cnt, _lock, _flag, f, _args, unpack):
    try:
        if unpack:
            _ret.append([_index, f(*_args)])
        else:
            _ret.append([_index, f(_args)])
    except Exception as e:
        _err.append(e)
    finally:
        with _lock:
            _cnt.value += 1
        _flag.value = 1


def _job_mp_chunk(_index, _ret, _err, _cnt, _lock, _flag, f, _args, unpack):
    try:
        ret = []
        for i in _args:
            if unpack:
                ret.append(f(*i))
            else:
                ret.append(f(i))
            with _lock:
                _cnt.value += 1
        _ret.append([_index, ret])
    except:
        _err.append(traceback.format_exc())
    finally:
        _flag.value = 1


def process_map(
    f,
    arr,
    num_workers=4,
    chunk_size=1,
    unpack=False,
    use_tqdm=True,
    tqdm_leave=True,
    shuffle=False,
    disable=False,
):
    if disable:
        it = tqdm(arr, dynamic_ncols=True, leave=tqdm_leave, disable=not use_tqdm)
        if unpack:
            return [f(*i) for i in it]
        return list(map(f, it))
    if chunk_size == 1:
        arr_ = list(enumerate(arr))
    else:
        arr_ = list(
            enumerate(arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size))
        )
    if shuffle:
        random.shuffle(arr_)
    with tqdm(
        total=len(arr),
        dynamic_ncols=True,
        leave=tqdm_leave,
        disable=not use_tqdm,
        miniters=0,
        smoothing=0,
    ) as bar:
        with mp.Manager() as mgr:
            last_cnt = 0
            cnt = mgr.Value("i", 0)
            ret = mgr.list()
            err = mgr.list()
            lock = mgr.Lock()
            ps = []
            flags = []
            target = _job_mp if chunk_size == 1 else _job_mp_chunk
            try:
                for i, args in arr_:
                    assert len(ps) <= len(flags)
                    if len(ps) == len(flags):
                        flag = mgr.Value("i", 0)
                        flags.append(flag)
                    else:
                        flag = flags[len(ps)]
                        flag.value = 0
                    p = mp.Process(
                        target=target,
                        args=(i, ret, err, cnt, lock, flag, f, args, unpack),
                    )
                    p.start()
                    ps.append([flag, p])
                    while len(ps) >= num_workers:
                        ps = [
                            [flag, p]
                            for flag, p in ps
                            if not flag.value or p.join() or p.close()
                        ]
                        bar.update(0)
                        assert not err
                        time.sleep(0.1)
                    c = cnt.value
                    bar.update(c - last_cnt)
                    last_cnt = c
                while cnt.value < len(arr):
                    c = cnt.value
                    bar.update(c - last_cnt)
                    last_cnt = c
                    assert not err
                    time.sleep(0.1)
                assert not err
            except AssertionError:
                print(err[0])
                exit()
            finally:
                for _, p in ps:
                    p.join()
            bar.update(len(arr) - last_cnt)
            ret = list(ret)
        ret.sort(key=lambda x: x[0])
    if chunk_size == 1:
        return [i for _, i in ret]
    else:
        return [j for _, i in ret for j in i]


def _test_1(i):
    # time.sleep(random.random() * 0.01)
    return i


def _test_2(i):
    time.sleep(random.random() * 0.1)
    return list(range(i * 10000))


def _test_3(i):
    time.sleep(random.random() * 0.1)
    assert random.random() > 0.1


def test():
    # ret_0 = thread_map(_test_1, range(100), disable=True)
    # ret_1 = thread_map(_test_1, range(100))
    ret_2 = process_map(_test_1, range(1000000), chunk_size=3)
    # assert ret_0 == ret_1 == ret_2
    # ret_1 = thread_map(_test_2, range(100))
    # ret_2 = process_map(_test_2, range(100))
    # assert ret_1 == ret_2
    # process_map(_test_3, range(100), chunk_size=2)


if __name__ == "__main__":
    test()
