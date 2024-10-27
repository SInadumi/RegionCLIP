import os
import pickle
import tempfile
import warnings

import cv2
import torch
import numpy as np
import logging
from tqdm import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_inputs(cfg, image):
    """Given a file name, return a list of dictionary with each dict corresponding to an image
    (refer to detectron2/data/dataset_mapper.py)
    """
    # image loading
    dataset_dict = {}
    dataset_dict["height"], dataset_dict["width"] = (
        image.shape[0],
        image.shape[1],
    )  # h, w before transforms

    # image transformation
    augs = utils.build_augmentation(cfg, False)
    augmentations = T.AugmentationList(
        augs
    )  # [ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333, sample_style='choice')]
    aug_input = T.AugInput(image)
    transforms = augmentations(aug_input)
    image = aug_input.image
    h, w = image.shape[:2]  # h, w after transforms
    dataset_dict["image"] = torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    )

    dataset_dict["scale_y"] = h / dataset_dict["height"]
    dataset_dict["scale_x"] = w / dataset_dict["width"]

    return [dataset_dict]


def create_model(cfg):
    """Given a config file, create a detector
    (refer to tools/train_net.py)
    """
    # create model
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    if (
        cfg.MODEL.META_ARCHITECTURE in ["CLIPRCNN", "CLIPFastRCNN", "PretrainFastRCNN"]
        and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None
        and cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN"
    ):  # load 2nd pretrained model
        DetectionCheckpointer(
            model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True
        ).resume_or_load(cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False)

    assert model.clip_crop_region_type == "RPN"
    assert model.use_clip_c4  # use C4 + resnet weights from CLIP
    assert model.use_clip_attpool  # use att_pool from CLIP to match dimension

    model.roi_heads.box_predictor.vis = (
        True  # get confidence scores before multiplying RPN scores, if any
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def extract_region_feats(cfg, model, batched_inputs):
    """Given a model and the input images, extract region features and save detection outputs into a local file
    (refer to detectron2/modeling/meta_arch/clip_rcnn.py)
    """
    im_id = 0  # single image

    # model inference
    # 1. localization branch: offline modules to get the region proposals
    images = model.offline_preprocess_image(batched_inputs)
    features = model.offline_backbone(images.tensor)
    proposals, _ = model.offline_proposal_generator(images, features, None)
    # visualize_proposals(batched_inputs, proposals, model.input_format)

    # 2. recognition branch: get 2D feature maps using the backbone of recognition branch
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)

    # 3. given the proposals, crop region features from 2D image features
    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = model.roi_heads._shared_roi_transform(
        [features[f] for f in model.roi_heads.in_features],
        proposal_boxes,
        model.backbone.layer4,
    )
    att_feats = model.backbone.attnpool(box_features)

    # 4. prediction head classifies the regions (optional)
    predictions = model.roi_heads.box_predictor(
        att_feats
    )  # predictions[0]: class logits; predictions[1]: box delta
    pred_instances, keep_indices = model.roi_heads.box_predictor.inference(
        predictions, proposals
    )  # apply per-class NMS
    results = model._postprocess(
        pred_instances, batched_inputs
    )  # re-scale boxes back to original image size

    # save detection outputs into files
    boxes = (
        results[im_id]["instances"].get("pred_boxes").tensor.cpu()
    )  # boxes after per-class NMS, [#boxes, 4]
    scores = (
        results[im_id]["instances"].get("scores").cpu()
    )  # scores after per-class NMS, [#boxes]
    classes = (
        results[im_id]["instances"].get("pred_classes").cpu()
    )  # class predictions after per-class NMS, [#boxes], class value in [0, C]

    # save features of detection regions (after per-class NMS)
    saved_dict = {}
    saved_dict["boxes"] = boxes
    saved_dict["scores"] = scores
    saved_dict["classes"] = classes

    return saved_dict


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as temp_dir:
        filename = os.path.join(temp_dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def main(args):
    cfg = setup(args)
    model = create_model(cfg)

    video = cv2.VideoCapture(args.video_input)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # basename = os.path.basename(args.video_input)
    codec, file_ext = (
        ("x264", ".mkv")
        if test_opencv_video_format("x264", ".mkv")
        else ("mp4v", ".mp4")
    )
    if codec == ".mp4v":
        warnings.warn("x264 codec not available, switching to mp4v", stacklevel=2)

    export_obj = []
    frame: np.ndarray  # (h, w, 3), BGR
    for frame in tqdm(_frame_from_video(video), total=num_frames):
        batched_inputs = get_inputs(cfg, frame)
        # extract region features
        with torch.no_grad():
            output = extract_region_feats(cfg, model, batched_inputs)
        # TODO: Check xywh or xyxy mode
        boxes, scores, classes = output["boxes"], output["scores"], output["classes"]
        chosen = (scores > args.confidence_threshold).nonzero()
        boxes, scores, classes = (
            boxes[chosen].reshape(-1, 4),
            scores[chosen].reshape(-1),
            classes[chosen].reshape(-1),
        )

        fields_list = []
        assert len(boxes) == len(scores) == len(classes)
        for box, score, _cls in zip(boxes, scores, classes):
            fields = (
                box[0],
                box[1],
                box[2],
                box[3],
                score,
                _cls,
            )  # x1, y1, x2, y2, score, class
            fields_list.append(fields)
        export_obj.append(np.array(fields_list))

    with open(args.output, mode="wb") as f:
        pickle.dump(export_obj, f)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--video-input",
        required=True,
        type=str,
        help="path to video file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="A file or directory to save output visualizations. ",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
