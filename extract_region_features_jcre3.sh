#!/usr/bin/env bash

set -euxo pipefail
readonly ROOT_DIR="path/to/data/jcre3"

MODEL_WEIGHTS=regionclip_pretrained-cc_rn50

poetry run python -u tools/extract_region_features.py \
    --root-dir ${ROOT_DIR} \
    --dataset-name jcre3 \
    --eval-gold \
    --output-file CLIP_fast_rcnn_R_50_C4_zsinf_w_GT \
    --config-file configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
    TEST.DETECTIONS_PER_IMAGE 64 \
    MODEL.WEIGHTS pretrained_ckpt/regionclip/${MODEL_WEIGHTS}.pth \
    MODEL.CLIP.TEXT_EMB_PATH pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
    MODEL.CLIP.CROP_REGION_TYPE RPN \
    MODEL.CLIP.MULTIPLY_RPN_SCORE True \
    MODEL.CLIP.OFFLINE_RPN_CONFIG configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS pretrained_ckpt/rpn/rpn_lvis_866.pth \
    SEED 10

poetry run python -u tools/extract_region_features.py \
    --root-dir ${ROOT_DIR} \
    --dataset-name jcre3 \
    --output-file CLIP_fast_rcnn_R_50_C4_zsinf \
    --config-file configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
    TEST.DETECTIONS_PER_IMAGE 64 \
    MODEL.WEIGHTS pretrained_ckpt/regionclip/${MODEL_WEIGHTS}.pth \
    MODEL.CLIP.TEXT_EMB_PATH pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
    MODEL.CLIP.CROP_REGION_TYPE RPN \
    MODEL.CLIP.MULTIPLY_RPN_SCORE True \
    MODEL.CLIP.OFFLINE_RPN_CONFIG configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS pretrained_ckpt/rpn/rpn_lvis_866.pth \
    SEED 10

echo "done!"
