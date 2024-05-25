#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=./DATA  # /path/to/datasets
TRAINER=CLIP_Adapter

DATASET=PACS
CFG=ViT_ep30
TEST_ENV='test_on_sketch.json'
NL=True

for SEED in 1 2 3
do
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir ./outputs/PACS/${TEST_ENV}/${TRAINER}_${CFG}/seed${SEED}/-1 \
DATASET.NUM_SHOTS -1 \
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
--model-dir ./outputs/PACS/${TEST_ENV}/${TRAINER}_${CFG}/seed${SEED}/-1 \
DATASET.NUM_SHOTS -1 \
TEST_ENV ${TEST_ENV}  \
VIZ.FeatDiv True
done

