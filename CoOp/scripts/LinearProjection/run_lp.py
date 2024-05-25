import os
import numpy as np
import random

CSC = 'False'
CTP = 'end'

for seed in [1]:
    for shots in [-1, 1, 2, 4, 8, 16]:
        for lr in  [1e-3, 1e-4, 1e-5, 1e-6]:
            os.system('CUDA_VISIBLE_DEVICES=0 bash lp_ccd.sh ColoredCatsDogs ViT_ep30 ' + str(seed) + ' ' + 'LinearProjection' + ' ' + str(shots))
            os.system('CUDA_VISIBLE_DEVICES=0 bash lp_vlcs.sh VLCS ViT_ep30 ' + str(seed) + ' ' + 'LinearProjection' + ' ' + str(shots))
            os.system('CUDA_VISIBLE_DEVICES=0 bash lp_pacs.sh PACS ViT_ep30 ' + str(seed) + ' ' + 'LinearProjection' + ' ' + str(shots))
            os.system('CUDA_VISIBLE_DEVICES=0 bash lp_celeba.sh CelebA ViT_ep30 ' + str(seed) + ' ' + 'LinearProjection' + ' ' + str(shots))

