import json
import pickle
import sys
from collections import defaultdict

import folium
from coord_convert.transform import wgs2gcj
from shapely.geometry import Point, Polygon
from tqdm import tqdm

sys.path.append("..")
from toputils import *


tgttop, _, config, projector = read_config(proj=True)
input_path = "../dataset/" + config["dataset"]["camera"]
output_path = f"data/r2cameras_{tgttop}.pkl"


def read():
    if tgttop == "sz1st":
        idx_lon, idx_lat, idx_name = 1, 2, 0
    else:
        idx_lon, idx_lat, idx_name = 7, 8, 3

    lines = open(input_path).readlines()[1:]
    print("input cameras:", len(lines))
    poly_rs = [
        [
            Polygon([projector(*p) for p in json.loads(v["bound"])]), 
            r
        ] 
        for r, v in config["regions"].items()
    ]
    r2cameras = defaultdict(list)
    not_in_any_region_cnt = 0
    cameras_out = []
    for i, l in enumerate(lines):
        l = l.split(",")
        lon, lat = float(l[idx_lon]), float(l[idx_lat])
        x, y = projector(lon, lat)
        p = Point(x, y)
        out_flag = True
        for poly, r in poly_rs:
            if p.covered_by(poly):
                r2cameras[r].append({
                    "id": i,
                    "name": l[idx_name],
                    "gps": (lon, lat),
                })
                out_flag = False
        if out_flag:
            not_in_any_region_cnt += 1
            cameras_out.append({
                    "id": i,
                    "name": l[idx_name],
                    "gps": (lon, lat),
                })
    pprint({k: len(v) for k, v in r2cameras.items()})
    print("not_in_any_region_cnt:", not_in_any_region_cnt)
    assert len(lines) - not_in_any_region_cnt == sum(
        len(r2cameras[r]) 
        for r in config["regions"] if "a" not in r
        )
    
    return r2cameras, cameras_out
    

def plot(r2cameras, cameras_out):
    m = get_base_map()
    colors = get_colors()
    for i, (r, cams) in enumerate(r2cameras.items()):
        color = colors[i % len(colors)]
        if "a" in r:
            continue
        for c in cams:
            folium.CircleMarker(
                # location=wgs2gcj(*c["gps"])[::-1],
                location=c["gps"][::-1],
                radius=3,
                popup=(c["id"], c["name"]),
                opacity=0.7,
                color=color,
            ).add_to(m)
        folium.PolyLine(
            locations=[p[::-1] for p in json.loads(config["regions"][r]["bound"])],
            popup=r,
            color=color
        ).add_to(m)
    
    for c in cameras_out:
        folium.CircleMarker(
            location=c["gps"][::-1],
            radius=3,
            popup=(c["id"], c["name"]),
            opacity=0.7,
            color="black",
        ).add_to(m)
    folium.PolyLine(
        locations=[p[::-1] for p in json.loads(config["bound"])],
        popup=tgttop,
        color="black"
    ).add_to(m)

    for n, v in config["regions"].items():
        if "a" in n:
            folium.PolyLine(
                locations=[p[::-1] for p in json.loads(v["bound"])],
                popup=tgttop,
                color="gray"
            ).add_to(m)

    m.save(f"figure/cameras_{tgttop}.html")


def main():
    region2cameras, cameras_out = read()

    plot(region2cameras, cameras_out)

    pickle.dump(region2cameras, open(output_path, "wb"))


if __name__ == "__main__":
    main()
