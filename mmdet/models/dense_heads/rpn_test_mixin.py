import sys
import torch
from mmdet.core import merge_aug_proposals

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class RPNTestMixin(object):
    """Test methods of RPN."""

    if sys.version_info >= (3, 7):

        async def async_simple_test_rpn(self, x, img_metas):
            sleep_interval = self.test_cfg.pop('async_sleep_interval', 0.025)
            async with completed(
                    __name__, 'rpn_head_forward',
                    sleep_interval=sleep_interval):
                rpn_outs = self(x)

            proposal_list = self.get_bboxes(*rpn_outs, img_metas)
            return proposal_list

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image.
        """
        rpn_outs = self(x)
        if torch.onnx.is_in_onnx_export():
            proposal_list = self.get_bboxes_onnx(*rpn_outs, img_metas)
        else:
            proposal_list = self.get_bboxes(*rpn_outs, img_metas)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas):
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, self.test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals


    def get_bboxes_onnx(self,
                    cls_scores,
                    bbox_preds,
                    img_metas,
                    cfg=None,
                    rescale=False,
                    ):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_anchors = torch.cat(mlvl_anchors)


        mlvl_cls_scores = [cls_scores[i].reshape((cls_scores[i].shape[0], cls_scores[i].shape[1], -1)) for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].reshape((bbox_preds[i].shape[0], bbox_preds[i].shape[1], -1)) for i in range(num_levels)]
        mlvl_cls_scores = torch.cat(mlvl_cls_scores, dim=2)
        mlvl_bbox_preds = torch.cat(mlvl_bbox_preds, dim=2)

        

        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        img_shapes = img_metas[0]['img_shape_for_onnx']

        result_list = self._get_bboxes_onnx(mlvl_cls_scores, mlvl_bbox_preds,
                                            mlvl_anchors, img_shapes,
                                            scale_factors, cfg, rescale,
                                            )
        return result_list

    def _get_bboxes_onnx(self,
                        mlvl_cls_scores,
                        mlvl_bbox_preds,
                        mlvl_anchors,
                        img_shapes,
                        scale_factors,
                        cfg,
                        rescale=False,
                        ):

            cfg = self.test_cfg if cfg is None else cfg
            # assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            #     mlvl_anchors)
            # batch_size = mlvl_cls_scores.shape[0]
            # convert to tensor to keep tracing
            # nms_pre_tensor = torch.tensor(
            #     cfg.get('nms_pre', -1),
            #     device=mlvl_cls_scores[0].device,
            #     dtype=torch.long)

            cls_score = mlvl_cls_scores.permute(0, 2, 1).reshape(mlvl_cls_scores.shape[0], -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                    scores = cls_score.sigmoid()
            else:
                    scores = cls_score.softmax(-1)
            bbox_pred = mlvl_bbox_preds.permute(0, 2, 1).reshape(mlvl_bbox_preds.shape[0], -1, 4)

            anchors = mlvl_anchors.expand_as(bbox_pred)

            

            # from mmdet.core.export import get_k_for_topk
            # nms_pre = get_k_for_topk(1000, bboxes.shape[1])

            max_scores, _ = scores.max(-1)
            _, topk_inds = max_scores.topk(1000)
            batch_inds = torch.arange(scores.shape[0]).view(
                scores.shape[0], 1).expand_as(topk_inds)  # 这个无法支持batch，只能支持batch==1的情况。
            anchors = anchors[batch_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]
    

            bboxes = self.bbox_coder.decode_onnx(anchors, bbox_pred, max_shape=img_shapes)

            # bboxes = bboxes.unsqueeze(2)
            


            # mlvl_bboxes = []
            # mlvl_scores = []
            # for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
            #                                          mlvl_bbox_preds,
            #                                          mlvl_anchors):

            #     assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            #     cls_score = cls_score.permute(0, 2, 3,
            #                                   1).reshape(batch_size, -1,
            #                                              self.cls_out_channels)
            #     if self.use_sigmoid_cls:
            #         scores = cls_score.sigmoid()
            #     else:
            #         scores = cls_score.softmax(-1)
            #     bbox_pred = bbox_pred.permute(0, 2, 3,
            #                                   1).reshape(batch_size, -1, 4)
            #     anchors = anchors.expand_as(bbox_pred)
                # # Always keep topk op for dynamic input in onnx
                # from mmdet.core.export import get_k_for_topk
                # nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
                # if nms_pre > 0:
                #     # Get maximum scores for foreground classes.
                #     if self.use_sigmoid_cls:
                #         max_scores, _ = scores.max(-1)
                #     else:
                #         # remind that we set FG labels to [0, num_class-1]
                #         # since mmdet v2.0
                #         # BG cat_id: num_class
                #         max_scores, _ = scores[..., :-1].max(-1)

                #     _, topk_inds = max_scores.topk(nms_pre)
                #     batch_inds = torch.arange(batch_size).view(
                #         -1, 1).expand_as(topk_inds)
                #     anchors = anchors[batch_inds, topk_inds, :]
                #     bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                #     scores = scores[batch_inds, topk_inds, :]

                # bboxes = self.bbox_coder.decode(
                #     anchors, bbox_pred, max_shape=img_shapes)
                # mlvl_bboxes.append(bboxes)
                # mlvl_scores.append(scores)

            # batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
            # if rescale:
            #     bboxes /= batch_mlvl_bboxes.new_tensor(
            #         scale_factors).unsqueeze(1)
            # batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

            # if self.use_sigmoid_cls:
            #     # Add a dummy background class to the backend when using sigmoid
            #     # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            #     # BG cat_id: num_class
            #     padding = batch_mlvl_scores.new_zeros(batch_size,
            #                                           batch_mlvl_scores.shape[1],
            #                                           1)
            #     batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

            return bboxes, scores