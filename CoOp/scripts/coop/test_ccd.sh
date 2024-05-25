#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=CoOp

DATASET=ColoredCatsDogs
CFG=ViT_ep30
TEST_ENV='ColoredCatsDogs.json'
NL=True

for SEED in 1 2 3
do
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir outputs/ColoredCatsDogs/ColoredCatsDogs.json/CoOp/seed${SEED}/ \
--load-epoch 30 \
TEST_ENV ${TEST_ENV}  \
VIZ.GRADCAM True
done