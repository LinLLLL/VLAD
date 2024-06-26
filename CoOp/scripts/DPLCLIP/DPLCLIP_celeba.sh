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

CTP=end
NCTX=16
CSC=False


for TEST_ENV in "test"
do
DIR=outputs/${DATASET}/${TEST_ENV}/${TRAINER}_${CFG}/seed${SEED}/${SHOTS}/
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
TRAINER.DPLCLIP.N_CTX ${NCTX} \
TRAINER.DPLCLIP.CSC ${CSC} \
TRAINER.DPLCLIP.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
TEST_ENV ${TEST_ENV}
fi
done