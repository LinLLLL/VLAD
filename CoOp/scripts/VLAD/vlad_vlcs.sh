#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=VLAD

DATASET=$1
CFG=$2  # config file
lambda1=$3
lambda2=$4
SEED=$5
SHOTS=$7
NL=True
knn=$6
ratio=$8
lr=$9


for TEST_ENV in "test_on_pascal.json"  "test_on_caltech.json"  "test_on_sun.json"  "test_on_labelme.json" 
do
DIR=outputs/${TRAINER}/${DATASET}_${CFG}_${lambda1}_${lambda2}_${knn}_${ratio}_${lr}/${TEST_ENV}/seed${SEED}/${SHOTS}
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
TRAINER.VLAD.non_linear_adapter ${NL} \
TRAINER.VLAD.lambda1 ${lambda1} \
TRAINER.VLAD.lambda2 ${lambda2} \
TRAINER.VLAD.knn ${knn} \
TRAINER.VLAD.ratio ${ratio} \
OPTIM.LR ${lr}
fi
done