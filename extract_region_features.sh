# Extract region features for a folder of images

#!/usr/bin/env bash

set -euxo pipefail
readonly DATA_DIR="path/to/J-CRe3"
readonly VIS_ANNOT_DIR="${DATA_DIR}/visual_annotations"
readonly RECORDING_DIR="${DATA_DIR}/recording"

MODEL_WEIGHTS=regionclip_pretrained-cc_rn50
OUTPUT_DIR="${DATA_DIR}/region_features/${MODEL_WEIGHTS}"
mkdir -p ${OUTPUT_DIR}

for run_id in $(cat ${DATA_DIR}/id/train.id ${DATA_DIR}/id/valid.id ${DATA_DIR}/id/test.id)
do
    echo "[ScenarioID: ${run_id}] Running object feature extraction"

    # NOTE: Configuration when using the poetry
    poetry run python tools/extract_region_features.py \
        --visual-path ${VIS_ANNOT_DIR}/${run_id}.json \
        --config-file configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
        --input-dir ${RECORDING_DIR}/${run_id}/images \
        --output-dir ${OUTPUT_DIR} \
        TEST.DETECTIONS_PER_IMAGE 30 \
        MODEL.WEIGHTS pretrained_ckpt/regionclip/${MODEL_WEIGHTS}.pth \
        MODEL.CLIP.TEXT_EMB_PATH pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
        MODEL.CLIP.CROP_REGION_TYPE RPN \
        MODEL.CLIP.MULTIPLY_RPN_SCORE True \
        MODEL.CLIP.OFFLINE_RPN_CONFIG configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
        MODEL.CLIP.BB_RPN_WEIGHTS pretrained_ckpt/rpn/rpn_lvis_866.pth \
        SEED 10
done
