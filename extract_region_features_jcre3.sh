#!/usr/bin/env bash

set -euxo pipefail
readonly ROOT_DIR="path/to/jcre3"
CONFIG_FILE="CLIP_fast_rcnn_R_50_C4_zsinf"
MODEL_WEIGHTS=regionclip_pretrained-cc_rn50

poetry run python -u tools/extract_region_features.py \
    --root-dir ${ROOT_DIR} \
    --dataset-name jcre3 \
    --output-file-name "${CONFIG_FILE}" \
    --config-file "configs/LVISv1-InstanceSegmentation/${CONFIG_FILE}.yaml" \
    TEST.DETECTIONS_PER_IMAGE 256 \
    MODEL.WEIGHTS pretrained_ckpt/regionclip/${MODEL_WEIGHTS}.pth \
    MODEL.CLIP.TEXT_EMB_PATH pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
    MODEL.CLIP.CROP_REGION_TYPE RPN \
    MODEL.CLIP.MULTIPLY_RPN_SCORE True \
    MODEL.CLIP.OFFLINE_RPN_CONFIG configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS pretrained_ckpt/rpn/rpn_lvis_866.pth \
    SEED 10

echo "done!"
