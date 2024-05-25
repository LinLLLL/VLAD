import os
import numpy as np
import random

for seed in [1, 2, 3]:
    for shots in [1, 2, 4, 8, 16, -1]:
        os.system('CUDA_VISIBLE_DEVICES=0 bash adapter_vlcs.sh VLCS ViT_ep30 ' + str(seed) + ' ' + 'CLIP_Adapter' + ' ' + str(shots))
        os.system('CUDA_VISIBLE_DEVICES=0 bash adapter_pacs.sh PACS ViT_ep30 ' + str(seed) + ' ' + 'CLIP_Adapter' + ' ' + str(shots))
        os.system('CUDA_VISIBLE_DEVICES=0 bash adapter_ccd.sh ColoredCatsDogs ViT_ep30 ' + str(seed) + ' ' + 'CLIP_Adapter' + ' ' + str(shots))
        os.system('CUDA_VISIBLE_DEVICES=0 bash adapter_celeba.sh CelebA ViT_ep30 ' + str(seed) + ' ' + 'CLIP_Adapter' + ' ' + str(shots))


