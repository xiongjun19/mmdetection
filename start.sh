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
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=6668 ./train.py  ~/soure_code/detection/mmdetection/myconfigs/yolov3_d53_608_273e_icdr2019.py --launcher > ./yolov3_d53_608_273e_icdr2019.log 2>$1 &

# nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=6668 ./tools/train.py  myconfigs/yolov3_d53_608_273e_icdr2019.py --launcher pytorch  > ./work_dirs/yolov3_d53_608_273e_icdr2019.log 2>&1 &
# nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=6668 ./tools/train.py  myconfigs/yolov3TD_d53_608_273e_icdr2019.py --launcher pytorch  > ./work_dirs/yolov3TD_d53_608_273e_icdr2019.log 2>&1 &

#nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=6668 ./tools/train.py  myconfigs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py --launcher pytorch  > ./work_dirs/faster_rcnn_r50_fpn_1x_coco.log 2>&1 &

#nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=6668 ./tools/train.py  myconfigs/yolov3/yolov3_resnet50_mstrain-608_12e_coco.py --launcher pytorch  > ./work_dirs/yolov3_resnet50_mstrain-608_12e_coco.log 2>&1 &


# nohup  python ./tools/train.py  myconfigs/yolov3/yolov3_resnet50_mstrain-608_12e_coco.py > ./work_dirs/yolov3_resnet50_mstrain-608_12e_coco.log 2>&1 &

# nohup  python ./tools/train.py  myconfigs/yolov3/yolov3_inceptionv1_mstrain_608_200e_coco.py > ./work_dirs/yolov3/yolov3_inceptionv1_mstrain_608_200e_coco.log 2>&1 &

#nohup  python ./tools/train.py  myconfigs/ssd/ssd_resnet34_512_24e_coco.py > ./work_dirs/ssd/ssd_resnet34_512_24e_coco.log 2>&1 &

# nohup  python ./tools/train.py  ./configs/ssd/ssd512_coco.py > ./work_dirs/ssd/ssd512_coco.log 2>&1 &
                   
#nohup  python ./tools/train.py  myconfigs/ssd/ssd_resnet34_512_120e_coco.py > ./work_dirs/ssd/ssd_resnet34_512_120e_coco.log 2>&1 &

# nohup  python ./tools/train.py  myconfigs/ssd/ssd_resnet34_512_100e_coco.py > ./work_dirs/ssd/ssd_resnet34_512_100e_coco.log 2>&1 &

# 
nohup  python ./tools/train.py  myconfigs/ssd/ssd_mobilenetv1_512_60e_coco.py > ./work_dirs/ssd/ssd_mobilenetv1_512_60e_coco.log 2>&1 &


