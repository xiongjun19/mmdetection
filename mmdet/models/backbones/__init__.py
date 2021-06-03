from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .trident_resnet import TridentResNet
from .inceptionv1 import GoogLeNet
from .inceptionv3 import Inception3
from .darknet_19 import Darknet_19
# import mmcls.models.backbones
# from mmcls.models.backbones.mobilenet_v2 import MobileNetV2
# from mmcls.models.backbones.mobilenet_v1 import MobileNetV1




__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet',
    'ResNeSt', 'TridentResNet', 'GoogLeNet', 'Inception3','Darknet_19',
    # 'MobileNetV2','MobileNetV1'
]
