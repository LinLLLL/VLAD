import os
import numpy as np
import random


for seed in [1. 2, 3]:
    random.seed(seed)
    knn = 10
    lr = 0.002
    ratio = 0.2
    for i in range(20):
        for shot in [1, 2, 4, 8, 16]:
            lambda_a = str(10 ** random.uniform(-3, -1))[:6]
            lambda_d = str(10 ** random.uniform(-2, 1))[:6]
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_vlcs.sh VLCS ViT_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_pacs.sh PACS ViT_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_vlcs.sh VLCS rn50_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_pacs.sh PACS rn50_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))


for seed in [1,2,3]:
    random.seed(seed)
    knn = 10
    lr = 0.002
    ratio = 0.2
    for i in range(20):
        for shot in [-1, 1, 2, 4, 8,16]:
            lambda_a = str(10 ** random.uniform(-1, 0))[:6]
            lambda_d = str(10 ** random.uniform(-1, 1))[:6]
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_ccd.sh ColoredCatsDogs ViT_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_celeba.sh CelebA ViT_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_ccd.sh ColoredCatsDogs rn50_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_celeba.sh CelebA rn50_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))
           


for seed in [1,2,3]:
    random.seed(seed)
    knn = 10
    lr = 0.002
    ratio = 0.2
    for i in range(1):
        for shot in [1, 4, 16]:
            lambda_a = "0.1"  # str(10 ** random.uniform(-1,0))[:6]
            lambda_d = "1.0"  # str(10 ** random.uniform(-1,1))[:6]
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_ccd.sh ColoredCatsDogs ViTL_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))
            lambda_a = "1.0"  # str(10 ** random.uniform(-1,0))[:6]
            lambda_d = "1.0"  # str(10 ** random.uniform(-1,1))[:6]
            os.system('CUDA_VISIBLE_DEVICES=0 bash vlad_celeba.sh CelebA ViTL_ep30 ' + ' ' + lambda_a + ' ' + lambda_d + ' ' + str(seed) + ' ' + str(knn) + ' ' + str(shot) + ' ' + str(ratio) + ' ' + str(lr))






            

