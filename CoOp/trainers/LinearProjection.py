import os.path as osp
import numpy as np
from scipy.stats import ortho_group
from sympy import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == 'ViT-L-14':
        model = torch.jit.load('Models/ViT-L-14.pt', map_location='cpu')
        model = clip.build_model(model.state_dict())
    elif backbone_name == 'ViT-B-16':
        print("****************Loading ViTB16 from openCLIP****************")
        model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(backbone_name, pretrained='laion400m_e31')
    else:         
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location='cpu').eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location='cpu')

        model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class Classifier(nn.Module):
    def __init__(self, nfeat, classnames):
        super().__init__()
        n_cls = len(classnames)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layer1 = nn.Sequential(
            nn.Linear(nfeat, int(nfeat)),
            nn.BatchNorm1d(int(nfeat)))
        self.layer2 = nn.Sequential(
           nn.Linear(int(nfeat), int(nfeat / 4)),
           nn.BatchNorm1d(int(nfeat / 4)))
        self.layer3 = nn.Sequential(
            nn.Linear(int(nfeat / 4), n_cls),
            nn.BatchNorm1d(n_cls))

    def forward(self, x):
        for layer in [self.layer1, self.layer2, self.layer3]:
            if layer == self.layer1:
                x1 = self.relu(layer(x))
            elif layer == self.layer2:
                x2 = self.relu(layer(x1))
            else:
                logits = (layer(x2))
        return [logits, x1]
        

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, domainnames, clip_model, nfeat):
        super().__init__()
        self.image_encoder_backbone = clip_model.visual
        self.classifier = Classifier(nfeat, classnames)

        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder_backbone(image.type(self.dtype))
        image_features = image_features.float()
        image_features.requires_grad_(requires_grad=True)
        output = self.classifier(image_features)

        return output

    def init_from_flat_params(self, init_params):
        cnt = 0
        for name, parameters in self.classifier.named_parameters():
            num = 1
            for i in range(len(list(parameters.shape))):
                num *= parameters.shape[i]
            parameters.data = init_params[cnt:cnt + num].reshape(list(parameters.shape)).type(
                parameters.dtype).to(device)
            cnt += num


@TRAINER_REGISTRY.register()
class LinearProjection(TrainerX):
   
    def check_cfg(self, cfg):
        assert cfg.TRAINER.LinearProjection.PREC in ["fp16", "fp32"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        domainnames = self.dm.dataset.alldomain

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LinearProjection.PREC == "fp32":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        if cfg.MODEL.BACKBONE.NAME == "RN50":
            nfeat = 1024
        elif cfg.MODEL.BACKBONE.NAME == "ViT-B/16":
            nfeat = 512
        else:
            nfeat = 768

        self.model = CustomCLIP(cfg, classnames, domainnames, clip_model, nfeat)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give linear projection classifier to the optimizer
        self.optim = build_optimizer(self.model.classifier, cfg.OPTIM)   
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("classifier", self.model.classifier, self.optim, self.sched)

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, domain_label, _, _ = self.parse_batch_train(batch)

        logits_category, image_features = self.model(image)
        loss = F.cross_entropy(logits_category, label)
        self.model_backward_and_update(loss)

        loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(logits_category, label)[0].item()
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        classname = batch["classname"]
        domainname = batch["domainname"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain, classname, domainname

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
