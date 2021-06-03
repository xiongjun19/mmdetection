
from torch.functional import norm
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential
from mmdet.core import bbox2result
from mmcv.cnn.bricks import ConvModule

# import numpy as np

@DETECTORS.register_module()
class SSDResNet(SingleStageDetector):

    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 512, 'S', 256, 128, 'S', 256, 128, 'S', 256),
        # 512: (256, 'S', 512, 512, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 input_size,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SSDResNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        assert input_size in (300, 512)
        self.input_size = input_size

        self.inplanes = 512
        self.extra = self._make_extra_layers(self.extra_setting[input_size])
        
        # self.strides=[1 ,1 ,2 ,2 ,2 ,2]
        # self.in_channels = bbox_head.in_channels
        # self._build_additional_features(38, self.in_channels)
        # self.extra = self.additional_blocks

        self._init_weights()

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

        # for i, l in enumerate(self.extra):
        #     x = l(x)
        #     outs.append(x)

        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
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
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                outs.append(x)
        outs = self.bbox_head(outs)
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
                conv = ConvModule(
                    self.inplanes, outplane, k, stride=2, padding=1, 
                    inplace=False, norm_cfg=dict(type='BN'))
                # conv = nn.Conv2d(
                #     self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = ConvModule(
                    self.inplanes, outplane, k, stride=1, padding=0, 
                    norm_cfg=dict(type='BN'), act_cfg=None)
                # conv = nn.Conv2d(
                #     self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        # if self.input_size == 512:
        #     layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return Sequential(*layers)

    def _build_additional_features(self, input_size, input_channels):
        idx = 0
        if input_size == 38:
            idx = 0
        elif input_size == 19:
            idx = 1
        elif input_size == 10:
            idx = 2

        self.additional_blocks = []

        if input_size == 38:
            self.additional_blocks.append(nn.Sequential(
                nn.Conv2d(input_channels[idx+1], 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, input_channels[idx+1], kernel_size=3, padding=1,stride=self.strides[2]),
                nn.ReLU(inplace=True),
            ))
            idx += 1

        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, input_channels[idx+1], kernel_size=3, padding=1, stride=self.strides[3]),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv9_1, conv9_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3, padding=1, stride=self.strides[4]),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv10_1, conv10_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=1,stride=self.strides[5]),
            nn.ReLU(inplace=True),
        ))
        idx += 1




        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):

        # layers = [
        #     *self.additional_blocks,
        #     ]

        # for layer in layers:
        #     for param in layer.parameters():
        #         if param.dim() > 1: nn.init.xavier_uniform_(param)

        for param in self.extra.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)
