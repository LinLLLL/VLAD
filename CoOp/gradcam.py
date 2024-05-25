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
from torch import autograd


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    fmap_block.append(output)


def hook_feature(out_):
    features.append(out_)


def cam_show_img(img, feature_map, grads, out_dir, out_dir1, name):
    H, W, _ = img.shape

    weights = np.sum(np.abs(grads), axis=0)
    cam = weights ** (1/6) # * feature_map.sum(0)

    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.resize(cam, (224, 224))
    img = cv2.resize(img, (224, 224))

    cam = cv2.boxFilter(cam, -1, (10, 10), normalize=True)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = heatmap

    path_cam_img = os.path.join(out_dir, name)
    path_cam_img1 = os.path.join(out_dir1, name)
    cv2.imwrite(path_cam_img, cam_img)
    cv2.imwrite(path_cam_img1, img)


def gradcam(algorithm, img_input, label, img_path):
    output_dir = /path/to/save/OriginalFigure  
    output_dir1 = /path/to/save/GradCamResults  

    algorithm.eval()  # 8

    for per_input, per_label, per_path in zip(img_input, label, img_path):
        fmap_block = list()
        grad_block = list()

        def backward_hook(module, grad_in, grad_out):
            grad_block.append(grad_out[0].detach())

        def farward_hook(module, input, output):
            fmap_block.append(output)

        for name, param in algorithm.named_parameters():
            param.requires_grad_(True)

        res = algorithm(per_input.unsqueeze(0))
        if len(res) == 3:
            pred, latent, text_feature = res
        else:
            pred = res[0]
        idx = np.argmax(pred.cpu().data.numpy())

        # backward
        algorithm.zero_grad()
        loss = F.cross_entropy(pred, per_label.unsqueeze(0))  # pred[0, 1]
        loss.backward()

        delta_x = 0.001 * torch.ones_like(per_input).cuda()
        delta_x.requires_grad_(True)
        if len(res) == 3:
            pred_delta = algorithm((delta_x + per_input).unsqueeze(0))[0] - pred
        else:
            pred_delta = algorithm((delta_x + per_input).unsqueeze(0))[0] - pred
        v_grad = autograd.grad(pred_delta.sum(), delta_x, create_graph=True)[0]

        grads_val = v_grad.cpu().data.numpy().squeeze()
        per_input.requires_grad_(False)

        if idx == per_label:
            res = 'TTT'
        else:
            res = 'FFF'

        img = cv2.imread(per_path, 1)
        cam_show_img(img, per_input.cpu().data.numpy().squeeze(), grads_val, output_dir, output_dir1, res + os.path.basename(per_path))



