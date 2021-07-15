import torch.nn as nn
# from mmcls.models.utils import make_divisible
from mmcv.cnn import ConvModule, constant_init, kaiming_init
# from .base_backbone import BaseBackbone
import logging
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.runner import _load_checkpoint


@BACKBONES.register_module()      
class MobileNetV1(BaseModule):
                  
    def __init__(self, 
                widen_factor,
                out_indices=(4, ),
                frozen_stages=-1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                norm_eval=False,
                with_cp=False,
                pretrained=None,
                ):
        super(MobileNetV1, self).__init__()


        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(32 * widen_factor, 8)
        self.conv1 = ConvModule(in_channels=3,out_channels=self.in_channels,
            kernel_size=3,stride=2,padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            )
        
        self.layers =[]
        
        layer_name = 'layer1'
        layer1 = nn.Sequential(
            self.make_dwconv_layer(32, 32, 3, 1),
            self.make_1x1conv_layer(32, 64, 1, 1),
            self.make_dwconv_layer(64, 64, 3, 2),
            self.make_1x1conv_layer(64, 128, 1, 1),
        )
        self.add_module(layer_name, layer1)
        self.layers.append(layer_name)

        layer_name = 'layer2'
        layer2 = nn.Sequential(
            
            self.make_dwconv_layer(128, 128, 3, 1),
            self.make_1x1conv_layer(128, 128, 1, 1),
            self.make_dwconv_layer(128, 128, 3, 2),
            self.make_1x1conv_layer(128, 256, 1, 1),
        )
        self.add_module(layer_name, layer2)
        self.layers.append(layer_name)

        layer_name = 'layer3'
        layer3 = nn.Sequential(
            
            self.make_dwconv_layer(256, 256, 3, 1),
            self.make_1x1conv_layer(256, 256, 1, 1),
            self.make_dwconv_layer(256, 256, 3, 2),
            self.make_1x1conv_layer(256, 512, 1, 1),
        )
        self.add_module(layer_name, layer3)
        self.layers.append(layer_name)

        layer_name = 'layer4'
        layer4 = nn.Sequential(
                        
            self.make_dwconv_layer(512, 512, 3, 1),
            self.make_1x1conv_layer(512, 512, 1, 1),
            self.make_dwconv_layer(512, 512, 3, 1),
            self.make_1x1conv_layer(512, 512, 1, 1),
            self.make_dwconv_layer(512, 512, 3, 1),
            self.make_1x1conv_layer(512, 512, 1, 1),
            self.make_dwconv_layer(512, 512, 3, 1),
            self.make_1x1conv_layer(512, 512, 1, 1),
            self.make_dwconv_layer(512, 512, 3, 1),
            self.make_1x1conv_layer(512, 512, 1, 1),


        )
        self.add_module(layer_name, layer4)
        self.layers.append(layer_name)

        layer_name = 'layer5'
        layer5 = nn.Sequential(

            self.make_dwconv_layer(512, 512, 3, 2),
            self.make_1x1conv_layer(512, 1024, 1, 1),            
        )
        self.add_module(layer_name, layer5)
        self.layers.append(layer_name)        

        layer_name = 'layer6'
        layer6 = nn.Sequential(
            
            self.make_dwconv_layer(1024, 1024, 3, 1),
            self.make_1x1conv_layer(1024, 1024, 1, 1)
        )
        self.add_module(layer_name, layer6)
        self.layers.append(layer_name)


        self.init_weights(pretrained)

        
        
    def make_dwconv_layer(self, in_channels, out_channels, kernel_size, stride):
        # in_channels, out_channels, kernel_size, stride = arch_setting
        in_channels = make_divisible(in_channels * self.widen_factor, 8)
        out_channels = make_divisible(out_channels * self.widen_factor, 8)

        return ConvModule(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=1,
                groups=in_channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
                )

    def make_1x1conv_layer(self, in_channels, out_channels, kernel_size, stride):
        # in_channels, out_channels, kernel_size, stride = arch_setting
        in_channels = make_divisible(in_channels * self.widen_factor, 8)
        out_channels = make_divisible(out_channels * self.widen_factor, 8)

        return ConvModule(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
                )
         

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
            self._init_weight(pretrained)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')
    def _init_weight(self, pretrained=None):
        # with torch.no_grad():
        checkpoint = _load_checkpoint(pretrained)
        if 'state_dict' in checkpoint:
            state_dict_pretrain = checkpoint['state_dict']
        else:
            state_dict_pretrain = checkpoint

        lst_pretrain_names = list(state_dict_pretrain)
        lst_pretrain_values = list(state_dict_pretrain.values())

        state_dict = self.state_dict()

        i =0
        for name, param in state_dict.items():
            if 'num_batches_tracked' not in name:
                param.copy_(lst_pretrain_values[i])
            i=i+1

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i+1 in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(MobileNetV1, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()



def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value