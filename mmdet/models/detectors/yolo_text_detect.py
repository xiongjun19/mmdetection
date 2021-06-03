# Copyright (c) 2019 Western Digital Corporation or its affiliates.

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
from mmdet.core import bbox2result
from mmdet.models.detectors.text_det_utils import TextDetector
import numpy as np

@DETECTORS.register_module()
class YOLOV3_TD(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOV3_TD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
        self.test_cfg = test_cfg
        self.text_cfg = test_cfg.text_cfg

    

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
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

        # bbox_results=list()
        # textdetector  = TextDetector(self.text_cfg.MAX_HORIZONTAL_GAP,
        #                                 self.text_cfg.MIN_V_OVERLAPS,
        #                                 self.text_cfg.MIN_SIZE_SIM)
        # for det_bboxes, det_labels in bbox_list:
        #     box_list = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            
        #     boxes = np.array(box_list[0][:,0:4],dtype=np.float32)
        #     scores = np.array(box_list[0][:,-1],dtype=np.float32)

        #     shape = img.shape[2:]
        #     boxes = textdetector.detect(boxes,
        #                         scores[:, np.newaxis],
        #                         shape,
        #                         self.text_cfg.TEXT_LINE_NMS_THRESH,
        #                         )

        #     bbox_results.append(boxes)
        
        return bbox_results

        # for mmocr output
        # results = dict(boundary_result=bbox_results)
        # return [results]
