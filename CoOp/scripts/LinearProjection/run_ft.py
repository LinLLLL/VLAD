import os
import numpy as np
import random

CSC = 'False'
CTP = 'end'

for seed in [1,2,3]:
    for shots in [-1, 1, 2, 4, 8, 16]:
        for lr in  [1e-3, 1e-4, 1e-5, 1e-6]:
            os.system('CUDA_VISIBLE_DEVICES=0 bash ft_ccd.sh ColoredCatsDogs ViT_ep30 ' + str(seed) + ' ' + 'LinearProjection_FT' + ' ' + str(shots) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash ft_celeba.sh CelebA ViT_ep30 ' + str(seed) + ' ' + 'LinearProjection_FT' + ' ' + str(shots) + ' ' + str(lr))

