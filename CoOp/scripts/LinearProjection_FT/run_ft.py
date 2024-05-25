import os
import numpy as np
import random

CSC = 'False'
CTP = 'end'

for seed in [1, 2, 3]:
    for shots in [16, -1]:
        for lr in  [1e-4, 1e-5, 1e-6, 1e-3]:
            os.system('CUDA_VISIBLE_DEVICES=0 bash ft_vlcs.sh VLCS ViT_ep200 ' + str(seed) + ' ' + 'LinearProjection_FT' + ' ' + str(shots) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash ft_pacs.sh PACS ViT_ep200 ' + str(seed) + ' ' + 'LinearProjection_FT' + ' ' + str(shots) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash ft_ccd.sh ColoredCatsDogs rn50_ep200 ' + str(seed) + ' ' + 'LinearProjection_FT' + ' ' + str(shots)  + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash ft_celeba.sh CelebA ViT_ep200 ' + str(seed) + ' ' + 'LinearProjection_FT' + ' ' + str(shots) + ' ' + str(lr))


