# Code for Dataset Generation and Validation

Here are codes for generation and validation of the dataset.

## File Description

- config.yml: specify the input paths and spatial parameters
- map_creator: get map from OSM and align cameras with map
- record_loader: pre-process the camera records data
- map_matcher: pre-process the historical GPS trajectory data and do map-matching
- speed_calculator: estimate road speed based on map-matched historical trajectory
- transition_calculator: estimate road transition based on map-matched historical trajectory
- main: vehicle Re-ID clustering, trajectory recovery, trajectory merging, dataset generation
  - config_sz3rd_cluster1.yml: config for main.py w.r.t one region
  - main.py: the core algorithm of iterative Re-ID clustering and trajectory recovery framework
    - clustering algorithm: sig_cluster.py
    - trajectory recovery model: routing.py
    - utils: pmap.py and utils.py
    - evaluating metrics: eval.py
  - merge_traj.py: merge the recovered trajectory across the divided regions
  - recover_traj.py: generate the final recovered trajectory based on the merged results
  - visualize_dataset.ipynb: visualize the statistics of the proposed dataset
  - get_csv.ipynb: convert the results to .csv files

## How We Run the Code

We sequentially perform following steps to generate this dataset:

- Specify the targeted city and the divided regions in config.yml
- map_creator
  - python read_camera.py
  - python get_map.py
  - python cal_shortest_path.py
- record_loader
  - python read_record.py
- map_matcher
  - python read_traj.py
  - python traj_match.py
- speed_calculator
  - python calc_speed.py
  - python train_speed.py
- transition_calculator
  - python calc_transition.py
- main
  - Specify the targeted city region in config.yml
  - python main.py
  - python merge_traj.py
  - python recover_traj.py
  - run visualize_dataset.ipynb
  - run get_csv.ipynb

Note that the codes for how we convert the raw video data to video records (cropped vehicle images with visual embeddings) are not provided here. Readers can refer to our previous work (cited in this paper in Methods->Vehicle re-identification->Vehicle Feature Extraction).
