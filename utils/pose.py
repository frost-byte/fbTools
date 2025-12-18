import os
import json
import warnings
import numpy as np
import comfy.model_management as model_management
from .images import make_empty_image, make_placeholder_tensor
try:
    from comfyui_controlnet_aux.src.custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
    from comfyui_controlnet_aux.src.custom_controlnet_aux.depth_anything import DepthAnythingDetector
    from comfyui_controlnet_aux.src.custom_controlnet_aux.midas import MidasDetector
    from comfyui_controlnet_aux.src.custom_controlnet_aux.zoe import ZoeDepthAnythingDetector, ZoeDetector
    from comfyui_controlnet_aux.src.custom_controlnet_aux.canny import CannyDetector
    from comfyui_controlnet_aux.src.custom_controlnet_aux.densepose import DenseposeDetector
    from comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose import OpenposeDetector
    from comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose import DwposeDetector
    from comfyui_controlnet_aux.utils import common_annotator_call
    _HAS_CONTROLNET_AUX = True
except Exception:
    _HAS_CONTROLNET_AUX = False
    pass

DWPOSE_MODEL_NAME = "yzd-v/DWPose"

#Trigger startup caching for onnxruntime
GPU_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider"]

def check_ort_gpu():
    try:
        import onnxruntime as ort
        for provider in GPU_PROVIDERS:
            if provider in ort.get_available_providers():
                return True
        return False
    except:
        return False

    if not os.environ.get("DWPOSE_ONNXRT_CHECKED"):
        if check_ort_gpu():
            print("DWPose: Onnxruntime with acceleration providers detected")
        else:
            warnings.warn("DWPose: Onnxruntime not found or doesn't come with acceleration providers, switch to OpenCV with CPU device. DWPose might run very slowly")
            os.environ['AUX_ORT_PROVIDERS'] = ''
        os.environ["DWPOSE_ONNXRT_CHECKED"] = '1'
        

def estimate_dwpose(
    image,
    detect_hand=True,
    detect_body=True,
    detect_face=True,
    resolution=512,
    bbox_detector="yolox_l.onnx",
    pose_estimator="dw-ll_ucoco_384.onnx",
    scale_stick_for_xinsr_cn=False,
    **kwargs
):
    if bbox_detector == "None":
        yolo_repo = DWPOSE_MODEL_NAME 
    elif bbox_detector == "yolox_l.onnx":
        yolo_repo = DWPOSE_MODEL_NAME
    elif "yolox" in bbox_detector:
        yolo_repo = "hr16/yolox-onnx"
    elif "yolo_nas" in bbox_detector:
        yolo_repo = "hr16/yolo-nas-fp16"
    else:
        raise NotImplementedError(f"Download mechanism for {bbox_detector}")

    if pose_estimator == "dw-ll_ucoco_384.onnx":
        pose_repo = DWPOSE_MODEL_NAME
    elif pose_estimator.endswith(".onnx"):
        pose_repo = "hr16/UnJIT-DWPose"
    elif pose_estimator.endswith(".torchscript.pt"):
        pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
    else:
        raise NotImplementedError(f"Download mechanism for {pose_estimator}")

    model = DwposeDetector.from_pretrained(
        pose_repo,
        yolo_repo,
        det_filename=(None if bbox_detector == "None" else bbox_detector), pose_filename=pose_estimator,
        torchscript_device=str(model_management.get_torch_device())
    )

    openpose_dicts = []
    def func(image, **kwargs):
        pose_img, openpose_dict = model(image, **kwargs)  # type: ignore
        openpose_dicts.append(openpose_dict)
        return pose_img

    out = common_annotator_call(
        func,
        image,
        include_hand=detect_hand,
        include_face=detect_face,
        include_body=detect_body,
        image_and_json=True,
        resolution=resolution,
        xinsr_stick_scaling=scale_stick_for_xinsr_cn
    )
    pose_json = json.dumps(openpose_dicts, indent=4)
    del model

    return out, pose_json

def dense_pose(
    image,
    model="densepose_r50_fpn_dl.torchscript",
    cmap="viridis",
    resolution=512,
):
    if not _HAS_CONTROLNET_AUX:
        return make_empty_image(height=resolution, width=resolution)

    denseModel = DenseposeDetector.from_pretrained(filename=model).to(model_management.get_torch_device())
    out = common_annotator_call(denseModel, image, cmap=cmap, resolution=resolution)
    del denseModel
    return out


def depth_anything_v2(
    image,
    ckpt="depth_anything_v2_vitl.pth",
    resolution=512,
):
    if not _HAS_CONTROLNET_AUX:
        return make_empty_image(height=resolution, width=resolution)

    depthAnyV2Model = DepthAnythingV2Detector.from_pretrained(filename=ckpt).to(model_management.get_torch_device())
    out = common_annotator_call(depthAnyV2Model, image, max_depth=1, resolution=resolution)
    del depthAnyV2Model
    return out

def depth_anything(
    image,
    ckpt="depth_anything_vitl14.pth",
    resolution=512,
):
    if not _HAS_CONTROLNET_AUX:
        return make_empty_image(height=resolution, width=resolution)

    depthAnyModel = DepthAnythingDetector.from_pretrained(filename=ckpt).to(model_management.get_torch_device())
    out = common_annotator_call(depthAnyModel, image, resolution=resolution)
    del depthAnyModel
    return out

def midas(
    image,
    a=np.pi * 2.0,
    bg_thresh=0.1,
):
    if not _HAS_CONTROLNET_AUX:
        return make_empty_image(height=image.shape[2], width=image.shape[3])

    midasModel = MidasDetector.from_pretrained().to(model_management.get_torch_device())
    out = common_annotator_call(midasModel, image, a=a, bg_th=bg_thresh)
    del midasModel
    return out

def openpose(
    image,
    include_hand=True,
    include_face=True,
    include_body=True,
    image_and_json=False,
    xinsr_stick_scaling=False,
    resolution=512,
):
    if not _HAS_CONTROLNET_AUX:
        return make_empty_image(height=resolution, width=resolution)

    openposeModel = OpenposeDetector.from_pretrained().to(model_management.get_torch_device())
    out = common_annotator_call(openposeModel, image, include_hand=include_hand, include_face=include_face, include_body=include_body, image_and_json=image_and_json, xinsr_stick_scaling=xinsr_stick_scaling, inresolution=resolution)
    del openposeModel
    return out

def zoe(
    image,
    resolution=512,
):
    if not _HAS_CONTROLNET_AUX:
        return make_empty_image(height=resolution, width=resolution)

    zoeModel = ZoeDetector.from_pretrained().to(model_management.get_torch_device())
    out = common_annotator_call(zoeModel, image, resolution=resolution)
    del zoeModel
    return out

def zoe_any(
    image,
    environment="indoor",
    resolution=512,
):
    if not _HAS_CONTROLNET_AUX:
        return make_empty_image(height=resolution, width=resolution)

    ckpt_name = "depth_anything_metric_depth_indoor.pt" if environment == "indoor" else "depth_anything_metric_depth_outdoor.pt"
    zoeAnyModel = ZoeDepthAnythingDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
    out = common_annotator_call(zoeAnyModel, image, resolution=resolution)
    del zoeAnyModel
    return out

def canny(
    image,
    low_threshold=100,
    high_threshold=200,
    resolution=512,
):
    if not _HAS_CONTROLNET_AUX:
        return make_empty_image(height=resolution, width=resolution)

    out = common_annotator_call(CannyDetector(), image, low_threshold=low_threshold, high_threshold=high_threshold, resolution=resolution)
    return out