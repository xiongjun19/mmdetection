#!/usr/bin/env bash

#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
    

GPUS=1
#PYTHONPATH=$PYTHONPATH:/home/ubuntu/source_code/detection/mmdetection
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

#nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=6668 ./train.py  ~/soure_code/detection/mmdetection/myconfigs/yolov3_d53_608_273e_icdr2019.py --launcher > ./yolov3_d53_608_273e_icdr2019.log 2>$1 &

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=6668 ./train.py  myconfigs/yolov3_d53_608_273e_icdr2019.py --launcher pytorch  > ./yolov3_d53_608_273e_icdr2019.log
