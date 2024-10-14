#!/usr/bin/env bash

set -euxo pipefail
readonly INPUT_ROOT="/mnt/home/shun-i/dev/cl_mmref/data/f30k_ent_jp"
readonly OUTPUT_ROOT="/mnt/home/shun-i/dev/cl_mmref/data/f30k_ent_jp"

MODEL_WEIGHTS=regionclip_pretrained-cc_rn50

poetry run python -u tools/extract_region_features.py \
    --input-root ${INPUT_ROOT} \
    --output-root ${OUTPUT_ROOT} \
    --dataset-name f30k_ent_jp \
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
