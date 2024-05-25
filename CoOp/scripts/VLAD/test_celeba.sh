#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=VLAD

DATASET=CelebA
CFG=ViT_ep30
TEST_ENV='test'
NL=True
DIR=/path/to/saved/model

for SEED in 1 2 3
do
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir ${DIR} \
--load-epoch 30 \
TRAINER.VLAD.non_linear_adapter ${NL} \
TEST_ENV ${TEST_ENV}  \
TRAINER.VLAD.ratio 0.5 \
VIZ.GRADCAM True
done