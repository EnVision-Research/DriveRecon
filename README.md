<<<<<<< HEAD
# DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving
### [Paper](https://arxiv.org/abs/0000.0000)  

> DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving

## Demo

![demo](./assets/s0.gif)


![demo](./assets/s6.gif)
![demo](./assets/s7.gif)
![demo](./assets/s8.gif) 

![demo](./assets/s1.gif)
![demo](./assets/s4.gif)
![demo](./assets/s5.gif)
## Getting Started

### Environmental Setups
Our code is developed on Ubuntu 22.04 using Python 3.9 
and pytorch=1.13.1+cu116. We also tested on pytorch=2.2.1+cu118. 
We recommend using conda for the installation of dependencies.

```bash
git clone https://github.com/EnVision-Research/DriveRecon.git --recursive
cd DriveRecon
conda env create -f environment.yml

pip install -r requirements.txt
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
@article{huang2024s3gaussian,
        title={DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving},
        author={Hao LU, Tianshuo XU, Wenzhao ZHENG, Yunpeng ZHANG, Wei ZHAN, Dalong DU, Masayoshi Tomizuka, Kurt Keutzer, Yingcong CHEN},
        journal={arXiv preprint arXiv:0000.0000},
        year={2024}
      }
```
=======
# DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving
### [Paper](https://arxiv.org/abs/0000.0000)  

> DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving



![vis](./assets/vis2.png)

## Demo

![demo](./assets/s0.gif)

## Getting Started

### Environmental Setups
Our code is developed on Ubuntu 22.04 using Python 3.9 and pytorch=1.13.1+cu116. We also tested on pytorch=2.2.1+cu118. We recommend using conda for the installation of dependencies.

```bash
git clone https://github.com/nnanhuang/S3Gaussian.git --recursive
cd S3Gaussian
conda create -n S3Gaussian python=3.9 
conda activate S3Gaussian

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

### Preparing Dataset
Follow detailed instructions in [Prepare Dataset](docs/prepare_data.md). 

We only use dynamic32 and static32 split.

### Training

For training first clip (eg. 0-50 frames), run 

```
python train.py -s $data_dir --port 6017 --expname "waymo" --model_path $model_path 
```

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{huang2024s3gaussian,
        title={DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving},
        author={},
        journal={arXiv preprint arXiv:0000.0000},
        year={2024}
      }
```
>>>>>>> b97c77629b784745f9cc19a0a061ff79feaefa94
