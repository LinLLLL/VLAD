import cv2
import os
import numpy as np
import torch
import json
import argparse
from sconf import Config
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import pandas as pd
from typing import Sequence

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import random


def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]  # 1
    img = np.ascontiguousarray(img)  # 2
    input_shape = (3, 224, 224)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((int(input_shape[1] / 0.875), int(input_shape[2] / 0.875))),
        transforms.CenterCrop(input_shape[1]),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])

    # transform = transforms.Compose(
    #     [
    #         T.Resize((224, 224)),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    img = transform(img)
    img = img.unsqueeze(0)  # 3
    return img


def compute_div(p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                eps_div: float, seed: int = 1) -> float:
    random.seed(seed)
    np.random.seed(seed)
    if not len(p) == len(q) == len(probs):
        raise ValueError
    div = 0
    for i in range(len(probs)):
        div += abs(p[i] - q[i]) / probs[i]
    div /= len(probs) * 2
    return div


def viz_feat(z_s, z_t, y_s, y_t, TRAINER, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(z_s.shape, z_t.shape)
    num_classes = len(set(y_s.tolist()))

    dn = []
    z = np.concatenate([z_s, z_t], 0)
    y = np.concatenate([y_s, y_t], 0)

    tsne = PCA(n_components=1, random_state=1)  # manifold.TSNE(n_components=1, random_state=1)  # 
    z = tsne.fit_transform(z)
    # print(z.shape)

    for i in range(num_classes):

        x_s = z[:z_s.shape[0]][np.where(y_s == i)]
        x_min, x_max = x_s.min(0), x_s.max(0)
        z_s0 = (x_s - x_min) / (x_max - x_min)

        x_t = z[z_s.shape[0]:][np.where(y_t == i)]
        x_min, x_max = x_t.min(0), x_t.max(0)
        z_t0 = (x_t - x_min) / (x_max - x_min)

        z_all = np.concatenate([z_s0, z_t0], 0)
        sampling_pdf = gaussian_kde(z_all.T)
        points = sampling_pdf.resample(1000, seed=seed)
        probs = sampling_pdf(points)

        p = gaussian_kde(z_s0.T)(points)
        q = gaussian_kde(z_t0.T)(points)
        dn.append(compute_div(p, q, probs, 1e-6, seed))
        print('distribution shift on class {}:'.format(str(i)), dn[-1])

        kernel_cs = gaussian_kde(z_s0.T)
        kernel_ct = gaussian_kde(z_t0.T)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 2, 1)
        x0 = np.arange(0, 1, 0.01)
        z0 = kernel_cs.evaluate(x0.T)
        ax.plot(x0, z0, color='blue')
        z0 = kernel_ct.evaluate(x0.T)
        ax.plot(x0, z0, color='red')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel("$Z$", fontsize=15)
        plt.legend(['train', 'test'], prop={'size': 16})

        plt.savefig('outputs/pdf_{}_Class{}.png'.format(TRAINER, i), bbox_inches='tight')

    print('add all distribution shifts:', np.sum(dn) / len(dn))







