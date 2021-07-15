
from mmcv.utils.config import DictAction
from mmdet.models import backbones
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential
from mmdet.core import bbox2result
# from torchvision.models.mobilenetv2 import MobileNetV2, mobilenet_v2
# from mmcls.models.backbones.mobilenet_v1 import MobileNetV1
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmcv.runner import _load_checkpoint
from mmcv.cnn.bricks import ConvModule
from torch.nn import ReLU

@DETECTORS.register_module()
class SSDMobileNetV1(BaseDetector):

    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S',256),
        512: (512, 'S', 512, 512, 'S', 256, 128, 'S', 256, 128, 'S', 256),
    }
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 input_size,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ssd_pretrained=None,
                 init_cfg=None,
                 ):
        super(SSDMobileNetV1, self).__init__(init_cfg)
        backbone.pretrained = pretrained
        # self.backbone = MobileNetV1(
        #     widen_factor=backbone.widen_factor,
        #     out_indices=backbone.out_indices,
        #     frozen_stages=backbone.frozen_stages,
        #     with_cp=backbone.with_cp,
        #     pretrained=backbone.pretrained,

        # )
        self.backbone = build_backbone(backbone)
        # self.backbone.init_weights(pretrained) #keys 不匹配，需要改进
        # chkpt = torch.load(pretrained)
        # self.backbone.load_state_dict(chkpt)
        # if isinstance(pretrained, str):
        #     self._init_backbone_weight(self.backbone, pretrained)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg        

        assert input_size in (300, 512)
        self.input_size = input_size

        self.inplanes = 1024
        self.extra = self._make_extra_layers(self.extra_setting[input_size])

        self.ssd_pretrained = ssd_pretrained
        
        # self.strides=[1 ,1 ,2 ,2 ,2 ,2]
        # self.in_channels = bbox_head.in_channels
        # self._build_additional_features(38, self.in_channels)
        # self.extra = self.additional_blocks
        # self._init_weights()
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        # super(SSDResNet, self).forward_train(img, img_metas)
        
        outs = list()
        out = self.extract_feat(img)
        out = list(out)
        x = out[-1]
        outs.extend(out)


        for i, layer in enumerate(self.extra):
            # x = F.relu(layer(x), inplace=True)
            x = layer(x)
            if i % 2 == 1:
                outs.append(x)

        losses = self.bbox_head.forward_train(outs, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses   

    def simple_test(self, img, img_metas, rescale=False):

        # x = self.extract_feat(img)
        outs = list()
        out = self.extract_feat(img)
        out = list(out)
        x = out[-1]
        outs.extend(out)
        for i, layer in enumerate(self.extra):
            # x = F.relu(layer(x), inplace=True)
            x =layer(x)
            if i % 2 == 1:
                outs.append(x)

        outs = self.bbox_head(outs)
        if torch.onnx.is_in_onnx_export():
            return outs
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

       
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        pass

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                # conv = ConvModule(
                #     self.inplanes, outplane, k, stride=2, padding=1, 
                #      norm_cfg=None)
                conv = Sequential(
                    nn.Conv2d(self.inplanes, outplane, k, stride=2, padding=1),
                    ReLU() # 主干网络中已经有一个了，这个就不需要了。
                ) 
                

            else:
                outplane = outplanes[i]
                # conv = ConvModule(
                #     self.inplanes, outplane, k, stride=1, padding=0, 
                #      norm_cfg=None)
                conv = Sequential(
                 nn.Conv2d(
                    self.inplanes, outplane, k, stride=1, padding=0),
                    ReLU()
                )

              
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        # if self.input_size == 512:
        #     layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return Sequential(*layers)

    # def _build_additional_features(self, input_size, input_channels):
    #     idx = 0
    #     if input_size == 38:
    #         idx = 0
    #     elif input_size == 19:
    #         idx = 1
    #     elif input_size == 10:
    #         idx = 2

    #     self.additional_blocks = []

    #     if input_size == 38:
    #         self.additional_blocks.append(nn.Sequential(
    #             nn.Conv2d(input_channels[idx+1], 512, kernel_size=1),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(512, input_channels[idx+1], kernel_size=3, padding=1,stride=self.strides[2]),
    #             nn.ReLU(inplace=True),
    #         ))
    #         idx += 1

    #     self.additional_blocks.append(nn.Sequential(
    #         nn.Conv2d(input_channels[idx], 256, kernel_size=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(256, input_channels[idx+1], kernel_size=3, padding=1, stride=self.strides[3]),
    #         nn.ReLU(inplace=True),
    #     ))
    #     idx += 1

    #     # conv9_1, conv9_2
    #     self.additional_blocks.append(nn.Sequential(
    #         nn.Conv2d(input_channels[idx], 128, kernel_size=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(128, input_channels[idx+1], kernel_size=3, padding=1, stride=self.strides[4]),
    #         nn.ReLU(inplace=True),
    #     ))
    #     idx += 1

    #     # conv10_1, conv10_2
    #     self.additional_blocks.append(nn.Sequential(
    #         nn.Conv2d(input_channels[idx], 128, kernel_size=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(128, input_channels[idx+1], kernel_size=1,stride=self.strides[5]),
    #         nn.ReLU(inplace=True),
    #     ))
    #     idx += 1




    #     self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def init_weights(self):
        if not self.ssd_pretrained:
            for param in self.extra.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)
        else:
            self.load_weight(self.ssd_pretrained)

    def load_weight(self, pretrained=None):
        # with torch.no_grad():
        checkpoint = _load_checkpoint(pretrained)
        if 'state_dict' in checkpoint:
            state_dict_pretrain = checkpoint['state_dict']
        else:
            state_dict_pretrain = checkpoint

        state_dict_pretrain_backbone = dict()
        state_dict_pretrain_extra = dict()
        state_dict_pretrain_head_cls = dict()
        state_dict_pretrain_head_reg = dict()
        lst_pretrain_names = list(state_dict_pretrain)
        for k in lst_pretrain_names:
            if 'classification_headers'in k:
                # k.replace('classification_headers', 'bbox_head.cls_convs', 1)
                state_dict_pretrain_head_cls[k.replace('classification_headers', 'bbox_head.cls_convs', 1)]  = state_dict_pretrain.pop(k)
            elif 'regression_headers' in k:
                # k.replace('regression_headers', 'bbox_head.reg_convs', 1)
                state_dict_pretrain_head_reg[k.replace('regression_headers', 'bbox_head.reg_convs', 1)]  = state_dict_pretrain.pop(k)
            elif 'extras' in k:
                # k.replace('extras', 'extra', 1)
                state_dict_pretrain_extra[k.replace('extras', 'extra', 1)]  = state_dict_pretrain.pop(k)
            elif 'base_net' in k:
                state_dict_pretrain_backbone[k.replace('base_net', 'back_bone',1)] = state_dict_pretrain.pop(k)   

        new_dict = dict(**state_dict_pretrain_backbone, **state_dict_pretrain_head_reg, **state_dict_pretrain_head_cls,  **state_dict_pretrain_extra )
        
        lst_pretrain_names = list(new_dict)
        lst_pretrain_values = list(new_dict.values())
        

        state_dict = self.state_dict()

        i =0
        for name, param in state_dict.items():
            if 'num_batches_tracked' not in name:
                param.copy_(lst_pretrain_values[i])
                i=i+1

 