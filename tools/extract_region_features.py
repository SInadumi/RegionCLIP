#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A script for region feature extraction
"""

import torch
import numpy as np
import json
import h5py
import logging
from pathlib import Path

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

from dataset_utils.annotation import ImageTextAnnotation

lvis_categories = json.load(open("./lvis_categories.json", "r", encoding="utf-8"))
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
    """Given a image return a list of dictionary with each dict corresponding to an image
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


def extract_region_feats(model, batched_inputs, gold_bbox=None):
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
        boxes = torch.tensor([list(x.rect.to_xyxy()) for x in gold_bbox]).to(
            model.device
        )
        # resize boxes
        scaled_boxes = Boxes(boxes)
        scaled_boxes.scale(
            scale_x=batched_inputs[im_id]["scale_x"],
            scale_y=batched_inputs[im_id]["scale_y"],
        )

        box_features = model.roi_heads._shared_roi_transform(
            [features[f] for f in model.roi_heads.in_features],
            [scaled_boxes],
            model.backbone.layer4,
        )
        att_feats = model.backbone.attnpool(box_features)  # region features

        boxes = boxes.cpu()
        scores = torch.tensor([1.0] * boxes.size()[0], dtype=torch.float32)
        classes = torch.tensor(
            [lvis_categories.index(x.className) for x in gold_bbox],
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

    visual_dir = Path(args.root_dir) / "image_text_annotation"
    image_root = Path(args.root_dir) / "recording"
    output_root = Path(args.root_dir)

    visual_paths = visual_dir.glob("*.json")
    image_ext = "png"
    if args.dataset_name == "f30k_ent_jp":
        image_ext = "jpg"
    with h5py.File(output_root / f"{args.output_file_name}.h5", mode="w") as output_fp:
        for source in visual_paths:
            scenario_id = source.stem
            logger.info(
                f"[ScenarioID: {scenario_id}] Running object feature extraction"
            )
            # print(f"[ScenarioID: {scenario_id}] Running object feature extraction")

            image_dir = image_root / scenario_id / "images"
            image_text_annotation = ImageTextAnnotation.from_json(
                Path(source).read_text()
            )
            image_files = [
                (image_dir / f"{image.imageId}.{image_ext}")
                for image in image_text_annotation.images
            ]

            # extract features per images
            for image_idx, img_file_name in enumerate(image_files):
                image_id = img_file_name.stem
                # image_id = img_file_name.stem
                image = utils.read_image(img_file_name, format="BGR")
                # extract region features
                with torch.no_grad():
                    batched_inputs = get_inputs(cfg, image)
                    output = extract_region_feats(cfg, model, batched_inputs)

                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/boxes", data=output["boxes"]
                )
                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/scores", data=output["scores"]
                )
                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/classes", data=output["classes"]
                )
                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/feats", data=output["feats"]
                )
                # NOTE: 配列としてアクセス可能
                # e.g.) output_fp[f"{scenario_id}/{image_id}/boxes"][0]


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--root-dir",
        required=True,
        type=str,
        help="path to input/output annotation dir",
    )
    parser.add_argument(
        "--dataset-name", required=True, type=str, choices=["jcre3", "f30k_ent_jp"]
    )
    parser.add_argument(
        "--output-file-name", required=True, type=str, default="default"
    )
    parser.add_argument("--eval-gold", action="store_true")
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
