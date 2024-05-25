#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=LinearProjection

DATASET=ColoredCatsDogs
CFG=ViT_ep200
TEST_ENV='ColoredCatsDogs.json'
NL=True
LR=$1
SHOTS=$2

for SEED in 1 2 3
do
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir outputs/ColoredCatsDogs/ColoredCatsDogs.json/LinearProjection/seed${SEED}/L${LR}_S${SHOTS} \
--load-epoch 200 \
DATASET.NUM_SHOTS -1 \
TEST_ENV ${TEST_ENV}  \
VIZ.FeatDiv True
done