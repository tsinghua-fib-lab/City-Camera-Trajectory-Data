# City-scale Vehicle Trajectory Data From Traffic Camera Videos

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)

Code and dataset of our dataset description paper:

- Fudan Yu, Huan Yan, Rui Chen, Guozhen Zhang, Meng Chen and Yong Li. City-scale Vehicle Trajectory Data From Traffic Camera Videos.

<!-- This is the official implementation of the following paper: 
- Fudan Yu, Wenxuan Ao, Huan Yan, Guozhen Zhang, Wei Wu and Yong Li. [Spatio-Temporal Vehicle Trajectory Recovery on Road Network Based on Traffic Camera Video Data(in KDD 2022)](https://dl.acm.org/doi/10.1145/3534678.3539186). 

<p align="center">
<img src=".\img\framework.png" height = "" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall Framework.
</p> -->

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

The .csv files as the proposed dataset.

### example

A .ipynb example on the basic usage of the proposed dataset.

<!-- ## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{10.1145/3534678.3539186,
author = {Yu, Fudan and Ao, Wenxuan and Yan, Huan and Zhang, Guozhen and Wu, Wei and Li, Yong},
title = {Spatio-Temporal Vehicle Trajectory Recovery on Road Network Based on Traffic Camera Video Data},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539186},
doi = {10.1145/3534678.3539186},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {4413–4421},
numpages = {9},
keywords = {spatio-temporal modeling, vehicle trajectory recovery, urban computing},
location = {Washington DC, USA},
series = {KDD '22}
}
``` -->