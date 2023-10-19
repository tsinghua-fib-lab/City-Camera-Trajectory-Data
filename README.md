# City-scale Vehicle Trajectory Data From Traffic Camera Videos

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)

Code of our dataset description paper:

- Fudan Yu, Huan Yan, Rui Chen, Guozhen Zhang, Meng Chen and Yong Li. [City-scale Vehicle Trajectory Data From Traffic Camera Videos (Scientific Data 2023)](https://doi.org/10.1038/s41597-023-02589-y). 

<p align="center">
<img src=".\img\framework.png" height = "" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall Framework.
</p>

## Requirements

- Python 3.7
- numpy == 1.21.3
- faiss == 1.5.3
- coloredlogs == 15.0.1
- osmnx == 1.1.2
- networkx == 2.6.3
- shapely == 1.8.0
- matplotlib == 3.4.3
- pyyaml == 6.0
- pandas == 1.3.4
- coord-convert == 0.2.1
- pytorch == 1.10.1

These dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

Another requirement ([Fast Map Matching for py3](https://github.com/John-Ao/fmm)) can be installed as follows:

- Install:

```bash
apt install libboost-dev libboost-serialization-dev gdal-bin libgdal-dev make cmake libbz2-dev libexpat1-dev swig python-dev
mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=~/.local && make -j32 && make install
```

- Test:

```bash
cd ../example/python && python fmm test.py
```

## File Description

### code

All the .py and .ipynb codes for how we generate the trajectory dataset based on visual embedded traffic camera records, evaluate the vehicle Re-ID and trajectory recovery metrics, and report statistical characteristic.

For details, see the README in this directory.

### dataset

The .csv files as the proposed dataset. Please download from our [Figshare repository](https://doi.org/10.6084/m9.figshare.c.6676199.v1).

### example

A .ipynb example on the basic usage of the proposed dataset.

## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@Article{Yu2023,
author={Yu, Fudan and Yan, Huan and Chen, Rui and Zhang, Guozhen and Liu, Yu and Chen, Meng and Li, Yong},
title={City-scale Vehicle Trajectory Data from Traffic Camera Videos},
journal={Scientific Data},
year={2023},
month={Oct},
day={17},
volume={10},
number={1},
pages={711},
issn={2052-4463},
doi={10.1038/s41597-023-02589-y},
url={https://doi.org/10.1038/s41597-023-02589-y}
}
```