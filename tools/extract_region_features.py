#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A script for region feature extraction
"""

import os
import torch
import numpy as np
import time
import json
import pathlib

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.structures import Boxes

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T

lvis_categories = json.load(open("./lvis_categories.json", "r", encoding="utf-8"))
lvis_categories.append(
    "region"
)  # J-CRe3における "className: 'region'" への対応．最終番地にインデックスを取る．


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


def extract_region_feats(cfg, model, batched_inputs, gold_bbox=None):
    """Given a model and the input images, extract region features and save detection outputs into a local file
    (refer to detectron2/modeling/meta_arch/clip_rcnn.py)
    """
    # model inference
    # 1. localization branch: offline modules to get the region proposals
    images = model.offline_preprocess_image(batched_inputs)
    features = model.offline_backbone(images.tensor)
    proposals, _ = model.offline_proposal_generator(images, features, None)
    # visualize_proposals(batched_inputs, proposals, model.input_format)

    # 2. recognition branch: get 2D feature maps using the backbone of recognition branch
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)

    if gold_bbox is None:
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
        im_id = 0  # single image
        boxes = (
            results[im_id]["instances"].get("pred_boxes").tensor.cpu()
        )  # boxes after per-class NMS, [#boxes, 4]
        scores = (
            results[im_id]["instances"].get("scores").cpu()
        )  # scores after per-class NMS, [#boxes]
        classes = (
            results[im_id]["instances"].get("pred_classes").cpu()
        )  # class predictions after per-class NMS, [#boxes], class value in [0, C]
        region_feats = att_feats[
            keep_indices[im_id]
        ].cpu()  # region features, [#boxes, d]

    else:
        boxes = torch.tensor([list(x["rect"].values()) for x in gold_bbox]).to(
            model.device
        )
        box_features = model.roi_heads._shared_roi_transform(
            [features[f] for f in model.roi_heads.in_features],
            [Boxes(boxes)],
            model.backbone.layer4,
        )
        att_feats = model.backbone.attnpool(box_features)  # region features

        boxes = boxes.cpu()
        scores = torch.tensor([1.0] * boxes.size()[0], dtype=torch.float32)
        classes = torch.tensor(
            [lvis_categories.index(x["className"]) for x in gold_bbox],
            dtype=torch.float32,
        )
        region_feats = att_feats.cpu()

    # save features of detection regions (after per-class NMS)
    saved_dict = {}
    saved_dict["boxes"] = boxes
    saved_dict["scores"] = scores
    saved_dict["classes"] = classes
    saved_dict["feats"] = region_feats

    return saved_dict


def main(args):
    cfg = setup(args)
    model = create_model(cfg)

    assert args.input_dir and args.output_dir
    assert args.visual_path

    # input images and annotations
    image_files = [(args.input_dir / x) for x in os.listdir(args.input_dir)]
    file_name = args.input_dir.parent.name
    visual_annotations = json.load(open(args.visual_path, "r", encoding="utf-8"))[
        "images"
    ]

    assert len(image_files) == len(visual_annotations)

    # extract features per images
    start = time.time()
    results = []
    for i, img_file_name in enumerate(image_files):
        res = []
        if i % 100 == 0:
            print("Used {} seconds for 100 images.".format(time.time() - start))
            start = time.time()

        # get input images and annots
        image = utils.read_image(img_file_name, format="BGR")
        batched_inputs = get_inputs(cfg, image)

        # extract region features
        with torch.no_grad():
            pred_dict = extract_region_feats(cfg, model, batched_inputs)

        # extract gold features
        # # [ {"imageId", "instanceId", "rect", "className"}, ... ]
        gold_bbox = visual_annotations[i]["boundingBoxes"]
        if len(gold_bbox) == 0:
            results.append(pred_dict)
            continue

        # extract region features
        with torch.no_grad():
            gold_dict = extract_region_feats(cfg, model, batched_inputs, gold_bbox)

        res = {
            k: torch.cat((v1, v2))
            for (k, v1), (_, v2) in zip(gold_dict.items(), pred_dict.items())
        }
        results.append(res)
    # save
    output_path = args.output_dir / f"{file_name}.pth"
    torch.save(results, output_path)

    print("done!")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--input-dir", type=pathlib.Path, default=None, help="Path to image directry"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="A scenario id file to save output bounding boxes",
    )
    parser.add_argument(
        "--visual-path",
        type=pathlib.Path,
        default=None,
        help="Path to J-CRe3 visual annotations",
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
