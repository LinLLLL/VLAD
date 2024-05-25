#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=./DATA  # /path/to/datasets
TRAINER=CLIP_Adapter

DATASET=ColoredCatsDogs
CFG=ViT_ep30
TEST_ENV='ColoredCatsDogs.json'

for SEED in 1 2 3
do
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir ./outputs/ColoredCatsDogs/CLIP_Adapter/seed${SEED}/-1/ \
--load-epoch 30 \
TEST_ENV ${TEST_ENV}  \
done


# To visualize the density curves of adapted image features 
for SEED in 1 2 3
do
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir ./outputs/ColoredCatsDogs/CLIP_Adapter/seed${SEED}/-1/ \
--load-epoch 30 \
TEST_ENV ${TEST_ENV}  \
VIZ.FeatDiv True
done