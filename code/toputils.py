import yaml
import json
import folium
from pprint import pprint
from pyproj import Proj


def read_config(proj=False):
    config = yaml.load(open("../config.yml", "r", encoding="utf8"), Loader=yaml.SafeLoader)
    target_top = config["target_top"]
    target = config["target"]
    config = config[target_top]
    results = [target_top, target, config]
    if proj:
        results.append(Proj(config["crs"]))
    return results


def get_colors():
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def get_base_map(gaode=False):
    lon, lat = json.loads(read_config()[-1]["center"])
    if gaode:
        m = folium.Map(
            location=[lat, lon],
            control_scale=True,
            tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
            attr='高德底图',
            zoom_start=13,
        )
    else:
        m = folium.Map(
            location=[lat, lon],
            control_scale=True,
            zoom_start=13,
        )
    m.add_child(folium.LatLngPopup())
    return m


def time_conventer(t):
    """将daytime转为hh:mm:ss格式"""
    t = round(t)
    assert 0 <= t < 86400
    h = t // 3600
    t -= h * 3600
    m = t // 60
    s = t - m * 60
    h = str(h) if h > 9 else f"0{h}"
    m = str(m) if m > 9 else f"0{m}"
    s = str(s) if s > 9 else f"0{s}"
    return ":".join([h, m, s])
