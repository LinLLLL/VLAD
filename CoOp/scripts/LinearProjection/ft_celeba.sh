#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets

DATASET=$1
CFG=$2  # config file
SEED=$3
TRAINER=$4
SHOTS=$5
LR=$6

CTP=end
NCTX=16
CSC=False
init_weights_path=outputs/CelebA/test/LinearProjection/seed${SEED}/${SHOTS}/classifier/model-best.pth.tar

for TEST_ENV in "test"
do
DIR=outputs/${DATASET}/${TEST_ENV}/${TRAINER}/seed${SEED}/lr${LR}_shots${SHOTS}/
if [ -d "$DIR" ]; then
echo "Results are available in ${DIR}. Skip this job"
else
echo "Run this job and save the output to ${DIR}"
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
TEST_ENV ${TEST_ENV} \
OPTIM.LR ${LR} \
MODEL.INIT_WEIGHTS ${init_weights_path}
fi
done