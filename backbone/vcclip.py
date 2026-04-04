import torch.nn as nn
import math
import copy
import torch
import torch.nn.functional as F


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'aircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'cars': 'a photo of a {}.',
    'food101': 'a photo of {}, a type of food.',
    'sun': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'ucf101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'imagenetr': 'a photo of a {}.',
    'cifar224': "a photo of a {}.",
    'cub': "a photo of a {}, a type of bird.",
    'objectnet': "a photo of a {}."
}


def load_clip_to_cpu(args):

    backbone_name = args["backbone_name"].lower()

    if backbone_name == 'clip_laion400m_e32':
        print('Using CLIP laion400m_e32 model as the backbone')
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16',
            pretrained='laion400m_e32'
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-16')

        return model, tokenizer


class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("I'm using Clip with a visual linear projection.")
        self.args = args
        self._device = args["device"][0]
        clip_model, tokenizer = load_clip_to_cpu(args)
        self.freeze(clip_model)
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post

        self.out_dim = 768

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, image):

        return self.frozen_forward(image)

    def frozen_forward(self, x):

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = self.transformer(x)

        x = self.ln_post(x[:, 0, :])

        return x