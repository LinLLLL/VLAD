import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'CelebA': 'a photo of a {}.',
    'VLCS': 'a photo of a {}.',
    'PACS': 'a photo of a {}.',
    'ColoredMNIST': 'a photo of a digit {}.',
    'ColoredCatsDogs': 'a photo of a {}.'
}



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


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, non_linear=False, TEXT_ENHANCED=False):
        super(Adapter, self).__init__()
        self.TEXT_ENHANCED = TEXT_ENHANCED
        if non_linear:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_in // reduction, bias=False),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(c_in // reduction, c_in, bias=False),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
            )

        else:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_in, bias=False)
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model, non_linear):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        n_feat = 1024 if cfg.MODEL.BACKBONE.NAME == "RN50" else 512
        n_feat = 768 if cfg.MODEL.BACKBONE.NAME == "ViT-L-14" else n_feat

        self.adapter = Adapter(n_feat, 4, non_linear).to(clip_model.dtype)

        self.ratio = cfg.TRAINER.VLAD.ratio

    def forward(self, image):
        text_features = self.text_encoder()

        image_features = self.image_encoder(image.type(self.dtype))

        x = self.adapter(image_features)
        ratio = self.ratio
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return [logits, x, text_features]


@TRAINER_REGISTRY.register()
class VLAD(TrainerX):
    """ VLAD """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_nei = cfg.TRAINER.VLAD.knn  

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model, cfg.TRAINER.VLAD.non_linear_adapter)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model('clip_adapter', self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output, latent, text_feature = self.model(image)
        loss_CE = F.cross_entropy(output, label)
        loss_al_d, loss_al_a = self.regularization(latent, text_feature, label)
        loss = loss_CE + self.cfg.TRAINER.VLAD.lambda1 * loss_al_d + self.cfg.TRAINER.VLAD.lambda2 * loss_al_a
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def regularization(self, latent, ek, y):
        distance = torch.mm(latent, latent.t())
        y = y.reshape(y.shape[0], -1)
        y1 = y.repeat(1, y.shape[0])
        y2 = y.t().repeat(y.shape[0], 1)
        lc = 1 * (y1 == y2)
        sort0 = torch.argsort(distance * lc, descending=True)

        index0 = torch.where((1 * (sort0 < self.n_nei)) & (lc==1))
        sort1 = torch.argsort(distance * (1 - lc), descending=True)
        index1 = torch.where((1 * (sort1 < self.n_nei)) & (lc == 0))
        pos = torch.sum(torch.exp(torch.diagonal(
            torch.mm(latent[index0[1]], ek[y[index0[0]]].squeeze(1).t())))) / len(index0[0])
        neg = torch.sum(torch.exp(torch.diagonal(
            torch.mm(latent[index1[1]], ek[y[index1[0]]].squeeze(1).t())))) / len(index0[0])
        neg1 = torch.sum(torch.exp(torch.mm(latent[index0[1]], ek.t()))) / len(index0[0])

        loss_al_d = -torch.log(pos / neg)
        loss_al_a = -torch.log(pos / neg1)

        return loss_al_d, loss_al_a

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
