source activate 2DGS
cd /vePFS001/luhao/Code/DriveRecon


## single gpu
#accelerate  launch --config_file ./acc.yaml  train.py  \
#--port 6017 --expname 'waymo' --configs 'arguments/nvs.py'

#### Multiple machines and multiple gps
#accelerate launch --config_file ./acc_config.yaml \
#--machine_rank $MLP_ROLE_INDEX --num_machines 3 --num_processes 24 \
#--main_process_ip $MLP_WORKER_0_HOST --main_process_port $MLP_WORKER_0_PORT  \
# train.py --port 6017 --expname 'waymo' \
#--configs 'arguments/nvs.py'

# evaling for reconstruction and novel view
# python eval.py --checkpoint_path "./checkpoint_10000.pth" --port 6017 --expname 'waymo' --configs 'arguments/nvs.py'

