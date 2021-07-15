import argparse
import os.path as osp
import warnings
import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv import DictAction
import cv2
from functools import partial
from mmdet.core.export import (build_model_from_cfg,
                               generate_inputs_and_wrap_model,
                               preprocess_example_input)

def generate_inputs_and_wrap_model_customized(config_path,
                                   checkpoint_path,
                                   input_config,
                                   cfg_options=None):


    model = build_model_from_cfg(
        config_path, checkpoint_path, cfg_options=cfg_options)
    one_img, one_meta = preprocess_example_input_customized(input_config)
    tensor_data = [one_img]
    model.forward = partial(
        model.forward, img_metas=[[one_meta]], return_loss=False)

    # pytorch has some bug in pytorch1.3, we have to fix it
    # by replacing these existing op
    opset_version = 11
    # put the import within the function thus it will not cause import error
    # when not using this function
    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(opset_version)

    return model, tensor_data


# keep ratio ==true
def preprocess_example_input_customized(input_config):
    input_path = input_config['input_path']
    input_shape = input_config['input_shape']
    ori_img = cv2.imread(input_path)
    ori_h,ori_w,_ = ori_img.shape
    ratio = min(input_shape[2]/ori_h, input_shape[3]/ori_w)
    

    one_img = np.zeros((input_shape[3], input_shape[2], input_shape[1]), dtype=np.uint8)
    resize_image = cv2.resize(ori_img, (int(ori_w * ratio), int(ori_h * ratio)))
    one_img[:resize_image.shape[0], :resize_image.shape[1], ...] = resize_image

    if 'normalize_cfg' in input_config.keys():
        normalize_cfg = input_config['normalize_cfg']
        mean = np.array(normalize_cfg['mean'], dtype=np.float32)
        std = np.array(normalize_cfg['std'], dtype=np.float32)
        to_rgb = normalize_cfg.get('to_rgb', True)
        one_img = mmcv.imnormalize(one_img, mean, std, to_rgb=to_rgb)

    # one_img = mmcv.imread(input_path)
    # one_img = mmcv.imresize(one_img, input_shape[2:][::-1])
    show_img = one_img.copy()

    one_img = one_img.transpose(2, 0, 1)
    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(
        True)
    (_, C, H, W) = input_shape
    one_meta = {
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'show_img': show_img,
    }

    return one_img, one_meta
def pytorch2onnx(config_path,
                 checkpoint_path,
                 input_img,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 normalize_cfg=None,
                 dataset='coco',
                 test_img=None,
                 do_simplify=False,
                 cfg_options=None,
                 dynamic_export=None):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }

    # prepare original model and meta for verifying the onnx model
    # orig_model = build_model_from_cfg(
    #     config_path, checkpoint_path, cfg_options=cfg_options)
    
    
    # one_img, one_meta = preprocess_example_input(input_config)
    one_img, one_meta = preprocess_example_input_customized(input_config)
    model, tensor_data = generate_inputs_and_wrap_model_customized(
        config_path, checkpoint_path, input_config, cfg_options=cfg_options)
    output_names = ['rois', 'roi_scors', 'fp0', 'fp1', 'fp2', 'fp3']
    if model.with_mask:
        output_names.append('masks')
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
            },
            'rois': {
                0: 'batch',
            },
            'roi_scores': {
                0: 'batch',
            },            
            'fp0': {
                0: 'batch',
            },
            'fp1': {
                0: 'batch',
            },  
            'fp2': {
                0: 'batch',
            },
            'fp3': {
                0: 'batch',
            },          
        }


    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        input_names=['input'],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes)

    # model.forward = orig_model.forward

    # get the custom op path
    ort_custom_op_path = ''
    try:
        from mmcv.ops import get_onnxruntime_op_path
        ort_custom_op_path = get_onnxruntime_op_path()
    except (ImportError, ModuleNotFoundError):
        warnings.warn('If input model has custom op from mmcv, \
            you may have to build mmcv with ONNXRuntime from source.')

    if do_simplify:
        from mmdet import digit_version
        import onnxsim

        min_required_version = '0.3.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnx-simplify>={min_required_version}'

        input_dic = {'input': one_img.detach().cpu().numpy()}
        onnxsim.simplify(
            output_file, input_data=input_dic, custom_lib=ort_custom_op_path)
    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        from mmdet.core import get_classes, bbox2result
        from mmdet.apis import show_result_pyplot

        model.CLASSES = get_classes(dataset)
        num_classes = len(model.CLASSES)
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        if dynamic_export:
            # scale up to test dynamic shape
            h, w = [int((_ * 1.5) // 32 * 32) for _ in input_shape[2:]]
            input_config['input_shape'] = (1, 3, h, w)
        if test_img is not None:
            input_config['input_path'] = test_img
        one_img, one_meta = preprocess_example_input(input_config)
        tensor_data = [one_img]

        # get pytorch output
        pytorch_results = model(tensor_data, [[one_meta]], return_loss=False)
        pytorch_results = pytorch_results[0]
        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        session_options = rt.SessionOptions()
        # register custom op for ONNX Runtime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        feed_input_img = one_img.detach().numpy()
        if dynamic_export:
            # test batch with two input images
            feed_input_img = np.vstack([feed_input_img, feed_input_img])
        sess = rt.InferenceSession(output_file, session_options)
        onnx_outputs = sess.run(None, {net_feed_input[0]: feed_input_img})
        output_names = [_.name for _ in sess.get_outputs()]
        output_shapes = [_.shape for _ in onnx_outputs]
        print(f'ONNX Runtime output names: {output_names}, \
            output shapes: {output_shapes}')
        # get last image's outputs
        onnx_outputs = [_[-1] for _ in onnx_outputs]
        ort_dets, ort_labels = onnx_outputs[:2]
        onnx_results = bbox2result(ort_dets, ort_labels, num_classes)
        if model.with_mask:
            segm_results = onnx_outputs[2]
            cls_segms = [[] for _ in range(num_classes)]
            for i in range(ort_dets.shape[0]):
                cls_segms[ort_labels[i]].append(segm_results[i])
            onnx_results = (onnx_results, cls_segms)
        # visualize predictions
        if show:
            show_result_pyplot(
                model, one_meta['show_img'], pytorch_results, title='Pytorch')
            show_result_pyplot(
                model, one_meta['show_img'], onnx_results, title='ONNXRuntime')

        # compare a part of result
        if model.with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res, p_res, rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show onnx graph and detection outputs')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--dataset', type=str, default='coco', help='Dataset name')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='mean value used for preprocess input data')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='variance value used for preprocess input data')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), '../../tests/data/color.jpg')

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    assert len(args.mean) == 3
    assert len(args.std) == 3

    normalize_cfg = {'mean': args.mean, 'std': args.std}

    # convert model to onnx file
    pytorch2onnx(
        args.config,
        args.checkpoint,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg,
        dataset=args.dataset,
        test_img=args.test_img,
        do_simplify=args.simplify,
        cfg_options=args.cfg_options,
        dynamic_export=args.dynamic_export)
