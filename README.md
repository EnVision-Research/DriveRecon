# DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving
### [Paper](https://arxiv.org/abs/2412.09043)  

> DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving

## Demo

<div style="display: grid; gap: 10px;">
  <!-- 第一行：单个视频 -->
  <div style="grid-column: 1 / -1; text-align: center;">
    <img src="./assets/s0.gif" alt="demo" style="max-width: 100%; height: auto;">
  </div>

  <!-- 第二行：三个视频 -->
  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
    <img src="./assets/s6.gif" alt="demo" style="max-width: 100%; height: auto;">
    <img src="./assets/s7.gif" alt="demo" style="max-width: 100%; height: auto;">
    <img src="./assets/s8.gif" alt="demo" style="max-width: 100%; height: auto;">
  </div>

  <!-- 第三行：三个视频 -->
  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
    <img src="./assets/s1.gif" alt="demo" style="max-width: 100%; height: auto;">
    <img src="./assets/s4.gif" alt="demo" style="max-width: 100%; height: auto;">
    <img src="./assets/s5.gif" alt="demo" style="max-width: 100%; height: auto;">
  </div>
</div>


## Getting Started

### Environmental Setups

```bash
git clone https://github.com/EnVision-Research/DriveRecon.git --recursive
cd DriveRecon
conda env create -f environment.yml

pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

### Preparing Dataset
Follow detailed instructions in [Prepare Dataset](docs/prepare_data.md). 


### Training

#### Single machines
```
accelerate  launch --config_file ./acc.yaml \
train.py  --port 6017 --expname 'waymo' --configs 'arguments/nvs.py'
```


#### Multiple machines and multiple gps
```
accelerate launch --config_file ./acc_config.yaml \
--machine_rank $MLP_ROLE_INDEX --num_machines 3 --num_processes 24 \
--main_process_ip $MLP_WORKER_0_HOST --main_process_port $MLP_WORKER_0_PORT  \
train.py --port 6017 --expname 'waymo' \
--configs 'arguments/nvs.py'
```

## Evaling 
```
# python eval.py --checkpoint_path "./checkpoint_10000.pth" --port 6017 --expname 'waymo' --configs 'arguments/nvs.py'
```


## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{Lu2024DrivingRecon,
        title={DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving},
        author={Hao LU, Tianshuo XU, Wenzhao ZHENG, Yunpeng ZHANG, Wei ZHAN, Dalong DU, Masayoshi Tomizuka, Kurt Keutzer, Yingcong CHEN},
        journal={arXiv preprint arXiv:2412.09043},
        year={2024}
      }
```
