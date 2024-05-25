#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=VLAD

DATASET=VLCS
CFG=ViT_ep30
TEST_ENV="test_on_pascal.json"  
NL=False
DIR=/path/to/saved/model

for SEED in 1
do
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir ${DIR} \
TRAINER.VLAD.non_linear_adapter ${NL} \
TEST_ENV ${TEST_ENV}  \
VIZ.FeatDiv True
done