# mmdetection 使用说明
## 本工程说明
本工程重度依赖开源的mmdection 框架， 具体参考 [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
本工程目前的分支和官方的保持一致； 目前我们自己使用的branch 是 lt_2.12， 
主要注意的地方：
1. 将yolov4 融合到框架中去了。
2. 是从官方的 v2.12.0 迁移过来。

## 框架安装
### 安装pytorch
```shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### 安装第三方依赖
```shell
pip install albumentations==0.5.2
```

### 安装 mmcv
```shell
pip install mmcv-full==1.3.4 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
```

### 最后安装本工程

```shell
cd path_to_project/
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## 框架使用文档
主要参考mmdection 官方文档， 具体地址：
 [https://mmdetection.readthedocs.io/en/latest/](https://mmdetection.readthedocs.io/en/latest/)

