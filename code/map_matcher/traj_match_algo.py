"""FastMapMatch路网匹配算法供调用"""
import os
import time

import osmnx as ox
from fmm import STMATCH, GPSConfig, Network, NetworkGraph, ResultConfig, STMATCHConfig
from tqdm import tqdm
from eviltransform import distance as gps_distance


def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    """匹配算法预生成.shp文件"""
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")
    
    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges) 
    # gdf_edges["fid"] = gdf_edges.id               # 坑: gdf_edges.id 和 原始G.edges的id已经不同
    gdf_edges["fid"] = list(range(len(gdf_edges)))  # 给edge重新编码自然索引，之后再映射回edge_id
    
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)


def st_match(G, traj_data, config, traj_csv_path, shp_path, mm_result_path):
    """
    匹配轨迹到路网
    """
    start_time = time.time()

    # step.1 将待匹配轨迹按匹配模型需求的一定格式存储为csv文件
    header = ["id", "geom", "timestamp"]
    with open(traj_csv_path, "w") as f:
        f.write(";".join(header) + "\n")
        cnt = 0
        for points in traj_data:
            gpss = ",".join(
                [" ".join([str(point[0]), str(point[1])]) for point in points]
            )
            wkt = "LINESTRING(" + gpss + ")"
            tms = ",".join([str(int(point[2])) for point in points])
            f.write(str(cnt) + ";" + wkt + ";" + tms + "\n")
            cnt = cnt + 1

    # step.2 生成匹配模型model
    if not os.path.exists(shp_path):
        save_graph_shapefile_directional(G, filepath=shp_path)
    network = Network(shp_path + "/edges.shp", "fid", "u", "v")
    graph = NetworkGraph(network)
    model = STMATCH(network, graph)
    print(f"start map-matching, pre-match use time: {time.time() - start_time:.3f}s")

    # step.3 设置匹配参数
    match_config = STMATCHConfig(config["k"], config["radius"], config["gps_error"])

    input_config = GPSConfig()
    input_config.file = traj_csv_path
    input_config.id = "id"
    # print(input_config.to_string())

    result_config = ResultConfig()
    result_config.file = mm_result_path
    result_config.output_config.write_error = True
    result_config.output_config.write_opath = True
    result_config.output_config.write_cpath = True
    result_config.output_config.write_pgeom = True
    result_config.output_config.write_offset = True
    result_config.output_config.write_mgeom = False
    result_config.output_config.write_tpath = False
    # print(result_config.to_string())

    # step.4 并行化map-matching
    print("FMM read from: ", traj_csv_path)
    status = model.match_gps_file(
        input_config, result_config, match_config, use_omp=True
    )  # use_omp=True表示并行化
    print("match done with following info: ")
    print(status)
    print("FMM write match result to: ", mm_result_path)


def match_post(G, mm_result_path, gps_1m):
    """
    通用后处理流程 对输出的.txt文件进一步进行解释, 输出有意义的匹配结果
    G: networkx.MultiDiGraph
    mm_result_path: str, matching model save result .txt path
    return:  list of { 'index': 这条轨迹对应input_path中的哪条轨迹的索引
                       'cpath': 用edge odrid表示的匹配路径
                       'cpath_id': 匹配前各个轨迹点对应到cpath中哪一条的索引
                       'pgeom': 每个轨迹点匹配后的坐标
                    }
    """

    # 逐行读取文件, 去除header, 以及如"0;;;;LINESTRING()"表示匹配失败的结果
    lines_output = [
        i for i in open(mm_result_path).readlines()[1:] if not i.endswith(";\n")
    ]

    # 从匹配时使用的自然索引映射回edge_id
    _, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    edges_true_id = []
    for u, v, k in gdf_edges.index:
        edges_true_id.append(G.edges[u, v, k]["id"])
    # for _, r in gdf_edges.iterrows():  # 视osmnx版本而定, 可能需要改为这段代码
    #     edges_true_id.append(G.edges[r["u"], r["v"], r["key"]]["id"])
    id_mapping_dict = {a: b for a, b in enumerate(edges_true_id)}

    ret = []
    for line in tqdm(lines_output):
        index, opath, error, offset, pgeom, cpath = line.strip().split(";")
        # opath 每个点匹配到的边
        # error 每对匹配前后轨迹点间的距离
        # pgeom 每个原始轨迹点匹配后的坐标  'LINESTRING(lon lat,lon lat)'
        # cpath 匹配路径 [roadid, roadid]
        assert cpath, line

        index = int(index)
        error = [float(x) / gps_1m for x in error.split(",")]  # 经纬度近似转换为m
        offset = [float(i) for i in offset.split(",")]
        pgeom = [[float(x) for x in point.split()] for point in pgeom[11:-1].split(",")]
        # 从匹配时使用的自然索引映射回edge_id
        opath = [id_mapping_dict[int(i)] for i in opath.split(",")]
        cpath = [id_mapping_dict[int(i)] for i in cpath.split(",")]

        # cpath去重
        cpath = cpath[:1] + [j for i, j in zip(cpath, cpath[1:]) if j != i]
        # 计算路-路转移关系
        ps = {}
        for i in range(len(cpath) - 1):
            a = cpath[i]
            for j in range(i + 1, len(cpath)):
                b = cpath[j]
                if (a, b) not in ps or j - i < len(ps[a, b]):
                    ps[a, b] = cpath[i + 1 : j + 1]

        # 根据opath和offset生成cpath
        cpath = [opath[0]]
        cpath_id = []
        orig_id = []
        stack = [[0, opath[0], offset[0]]]
        assert len(opath) == len(offset)
        for i, (j, k) in enumerate(zip(opath, offset)):
            if j == stack[-1][1]:
                while stack and k <= stack[-1][2]:
                    stack.pop()
                stack.append([i, j, k])
            else:
                cpath_id += [len(cpath) - 1] * len(stack)
                orig_id += [i[0] for i in stack]
                cpath += ps[stack[-1][1], j]
                stack = [[i, j, k]]
        cpath_id += [len(cpath) - 1] * len(stack)
        orig_id += [i[0] for i in stack]
        # print(1 - len(orig_id) / len(opath))

        ret.append(
            {
                "index": index,
                "cpath": cpath,
                "cpath_id": cpath_id,
                "orig_id": orig_id,
                "error": error,
                "pgeom": pgeom,
            }
        )
    return ret


def remove_small_circle(
    edges,
    nodes,
    Circle_Size,
    Circle_Length=None,
    nodeid_to_gps=None,
    do_not_remove_tag=True,
):
    """
    去除route中小圈的算法
    匹配结果常在诸如井字形的多条邻近道路处直线通过时，变成绕一圈再通过，因此考虑手动去除
    edges: list of [ edgeid, [] ], 当[]不为空时, 表示这条edge不能被删去的标记
    nodes: list of 边的端点id (nodes_num = edges_num + 1)
    Circle_Size: num of edges of a circle to be removed
    Circle_Length: length of a circle to be removed
    nodes_GPS_dict: 从nodeid查gps
    do_not_remove_tag: 为False时, [ edgeid, [] ]当[]不为空时, 也可以被去除
    """
    if Circle_Length is not None:  # 除了用Circle_Size对边数来筛选小圈外, 还用Circle_Length圈的长度来筛选小圈
        flag = 0
        while flag == 0:
            last_visit = {}  # 记录点在前多少步被访问过
            for i, node in enumerate(nodes):
                if node in last_visit.keys():  # 检测到曾经到达过的node
                    tmp = last_visit[node]  # 几步之前曾经到达(圈的边数)

                    if tmp < Circle_Size:  # 边数为Circle_Size或更小的圈将被去除

                        points_circle = []  # 进一步检查该圈的路径长度，若较短则去除
                        for j in range(i - tmp, i + 1):
                            points_circle.append(nodeid_to_gps[nodes[j]])
                        circle_length = 0
                        num = len(points_circle)
                        for j in range(num):
                            [lon1, lat1] = points_circle[j]
                            [lon2, lat2] = points_circle[(j + 1) % num]
                            circle_length = circle_length + gps_distance(
                                lat1, lon1, lat2, lon2
                            )
                        if circle_length < Circle_Length:

                            to_be_del = edges[i - tmp - 1 : i]
                            can_be_del = True  # 进一步检查edge是否可以被去除的标记, 若都可以去除才去除
                            if do_not_remove_tag:
                                for edge in to_be_del:
                                    if edge[1]:
                                        can_be_del = False
                                        break
                            if can_be_del:
                                del nodes[i - tmp : i + 1]
                                del edges[i - tmp - 1 : i]
                                break  # 去除一个圈后，重新从头开始遍历

                # 若该node未曾到达，或者不符合去除圈的条件，则继续往下遍历，更新last_visit
                for key in last_visit.keys():
                    last_visit[key] = last_visit[key] + 1
                last_visit[node] = 0

                if i == len(nodes) - 1:  # 若遍历到最后一个点，算法结束
                    flag = 1

    else:  # 只用Circle_Size对边数来筛选小圈, 不考虑Circle_Length, 以避免距离计算的耗时

        flag = 0
        while flag == 0:
            last_visit = {}  # 记录点在前多少步被访问过
            for i, node in enumerate(nodes):
                if node in last_visit.keys():  # 检测到曾经到达过的node
                    tmp = last_visit[node]  # 几步之前曾经到达(圈的边数)

                    if tmp < Circle_Size:  # 边数为Circle_Size或更小的圈将被去除

                        to_be_del = edges[i - tmp - 1 : i]
                        can_be_del = True  # 进一步检查edge是否可以被去除的标记, 若都可以去除才去除
                        if do_not_remove_tag:
                            for edge in to_be_del:
                                if edge[1]:
                                    can_be_del = False
                                    break
                        if can_be_del:
                            del nodes[i - tmp : i + 1]
                            del edges[i - tmp - 1 : i]
                            break  # 去除一个圈后，重新从头开始遍历

                # 若该node未曾到达，或者不符合去除圈的条件，则继续往下遍历，更新last_visit
                for key in last_visit.keys():
                    last_visit[key] = last_visit[key] + 1
                last_visit[node] = 0

                if i == len(nodes) - 1:  # 若遍历到最后一个点，算法结束
                    flag = 1
    return edges


def remove_adj_same_road(edges):
    """
    原始route中, 不应出现连续两条相同道路, 但经过以上remove_circle的操作, 可能导致出现这种情况, 因此需修复
    """
    # edges: list of [ edgeid, [] ], 相邻相同道路将被合并, 此时[]中的内容也将被合并
    len_edges = len(edges)
    to_be_del = []  # 记录待删road的索引
    to_be_merge = []  # 记录待合并road的索引
    to_be_merge_result = []
    i = 0
    while True:
        if i == len_edges - 1:  # 到达末尾, 则退出循环
            break

        road_tmp = edges[i][0]
        for j in range(i + 1, len_edges):  # 找到之后第一条不与i相同的路为j(当最后一条路也与i相同时,则j有可能与i相同)
            if road_tmp != edges[j][0]:
                break

        if j == len_edges - 1:  # 当j为最后一条路时, 则j可能与i相同
            if edges[i][0] == edges[j][0]:  # 从i开始,直到最后一条路,都相同,则做如下处理
                cnt = 0
                for k in range(i, len_edges):
                    if edges[k][1] == []:  # 没有station对应的路可以删掉
                        to_be_del.append(k)
                    else:
                        cnt = cnt + 1  # 在这一段相同的路中,有多少条是有station对应的
                if cnt == 0:
                    to_be_del.pop()  # 如果全部被删去, 则保留最后一条
                if cnt > 1:  # 如果在这一段相同的路中,有station对应的路多于一条
                    one_merge_stations = []
                    for k in range(
                        i, len_edges
                    ):  # 将这些road合并, 合并后的road对应合并前所有road的station
                        to_be_merge.append(k)
                        one_merge_stations = one_merge_stations + edges[k][1]
                    one_merge_edge = [edges[i][0], one_merge_stations]
                    to_be_merge_result.append(
                        {"start": i, "end": len_edges, "merged_edge": one_merge_edge}
                    )
                break  # 处理完后退出循环

        # 排除以上情形后, 可确保j是之后第一条不与i相同的路
        if j == i + 1:  # 若下一条路就与i不同, 则不做处理
            i = j
            continue

        # 下面处理range(i,j)这段相同的路, 处理同上
        cnt = 0
        for k in range(i, j):  # 没有station对应的路可以删掉
            if edges[k][1] == []:
                to_be_del.append(k)
            else:
                cnt = cnt + 1  # 在这一段相同的路中,有多少条是有station对应的
        if cnt == 0:
            to_be_del.pop()  # 如果全部被删去, 则保留最后一条
        if cnt > 1:  # 如果在这一段相同的路中,有station对应的路多于一条
            one_merge_stations = []
            for k in range(i, j):  # 将这些road合并, 合并后的road对应合并前所有road的station
                to_be_merge.append(k)
                one_merge_stations = one_merge_stations + edges[k][1]
            one_merge_edge = [edges[i][0], one_merge_stations]
            to_be_merge_result.append(
                {"start": i, "end": j, "merged_edge": one_merge_edge}
            )

        i = j  # 跳到之后第一条不与这条路相同的路

    edges_processed = []
    merge_cnt = 0
    i = 0
    while i < len(edges):
        if i in to_be_merge:
            j = to_be_merge_result[merge_cnt]["end"]
            merged_edge = to_be_merge_result[merge_cnt]["merged_edge"]
            edges_processed.append(merged_edge)
            merge_cnt = merge_cnt + 1
            i = j
            continue
        if not i in to_be_del:
            edges_processed.append(edges[i])
        i = i + 1

    return edges_processed


def remove_adj_same_road_order(edges):
    """
    合并的相邻道路按order字段排序
    """
    # edges: list of [ edgeid, [] ], 相邻相同道路将被合并, 此时[]中的内容也将被合并
    len_edges = len(edges)
    to_be_del = []  # 记录待删road的索引
    to_be_merge = []  # 记录待合并road的索引
    to_be_merge_result = []
    i = 0
    while True:
        if i == len_edges - 1:  # 到达末尾, 则退出循环
            break

        road_tmp = edges[i][0]
        for j in range(i + 1, len_edges):  # 找到之后第一条不与i相同的路为j(当最后一条路也与i相同时,则j有可能与i相同)
            if edges[j][0] != road_tmp:
                break

        if j == len_edges - 1:  # 当j为最后一条路时, 则j可能与i相同
            if edges[i][0] == edges[j][0]:  # 从i开始,直到最后一条路,都相同,则做如下处理
                cnt = 0
                for k in range(i, len_edges):
                    if edges[k][1] == []:  # 没有station对应的路可以删掉
                        to_be_del.append(k)
                    else:
                        cnt = cnt + 1  # 在这一段相同的路中,有多少条是有station对应的
                if cnt == 0:
                    to_be_del.pop()  # 如果全部被删去, 则保留最后一条
                if cnt > 1:  # 如果在这一段相同的路中,有station对应的路多于一条
                    one_merge_stations = []
                    for k in range(
                        i, len_edges
                    ):  # 将这些road合并, 合并后的road对应合并前所有road的station
                        to_be_merge.append(k)
                        one_merge_stations = one_merge_stations + edges[k][1]
                    one_merge_stations.sort(key=lambda x: x["order"])
                    one_merge_edge = [edges[i][0], one_merge_stations]
                    to_be_merge_result.append(
                        {"start": i, "end": len_edges, "merged_edge": one_merge_edge}
                    )
                break  # 处理完后退出循环

        # 排除以上情形后, 可确保j是之后第一条不与i相同的路
        if j == i + 1:  # 若下一条路就与i不同, 则不做处理
            i = j
            continue

        # 下面处理range(i,j)这段相同的路, 处理同上
        cnt = 0
        for k in range(i, j):  # 没有station对应的路可以删掉
            if edges[k][1] == []:
                to_be_del.append(k)
            else:
                cnt = cnt + 1  # 在这一段相同的路中,有多少条是有station对应的
        if cnt == 0:
            to_be_del.pop()  # 如果全部被删去, 则保留最后一条
        if cnt > 1:  # 如果在这一段相同的路中,有station对应的路多于一条
            one_merge_stations = []
            for k in range(i, j):  # 将这些road合并, 合并后的road对应合并前所有road的station
                to_be_merge.append(k)
                one_merge_stations = one_merge_stations + edges[k][1]
            one_merge_stations.sort(key=lambda x: x["order"])
            one_merge_edge = [edges[i][0], one_merge_stations]
            to_be_merge_result.append(
                {"start": i, "end": j, "merged_edge": one_merge_edge}
            )

        i = j  # 跳到之后第一条不与这条路相同的路

    edges_processed = []
    merge_cnt = 0
    i = 0
    while i < len_edges:
        if i in to_be_merge:
            j = to_be_merge_result[merge_cnt]["end"]
            merged_edge = to_be_merge_result[merge_cnt]["merged_edge"]
            edges_processed.append(merged_edge)
            merge_cnt = merge_cnt + 1
            i = j
            continue
        if not i in to_be_del:
            edges_processed.append(edges[i])
        i = i + 1
    return edges_processed


if __name__ == "__main__":
    a = [
        [0, []],
        [0, [{"order": 1}]],
        [0, [{"order": 0}]],
        [1, []],
        [1, []],
        [2, []],
        [3, []],
        [4, [{"order": 1}]],
        [4, [{"order": 0}, {"order": 2}]],
    ]
    print(a)
    for x in remove_adj_same_road_order(a):
        print(x)
