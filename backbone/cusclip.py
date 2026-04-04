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

        model.out_dim = 512
        return model, tokenizer

class myCLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("I'm using Clip with a visual linear projection.")
        self.args = args
        self._device = args["device"][0]
        model, tokenizer = load_clip_to_cpu(args)
        self.freeze(model)
        self.image_encoder = model.visual
        self.text_encoder = model.encode_text
        self.tokenizer = tokenizer
        self.logit_scale = model.logit_scale
        self.out_dim = model.out_dim
        self.visual_adapter = None
        self.register_buffer('text_features', None)
        self.text_forward(args["shuffled_cls_names"])
        self.configure_new_task()

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def configure_new_task(self):
        visual_adapter = nn.Linear(self.out_dim, self.out_dim, bias=False)
        visual_adapter.requires_grad_(True)
        self.visual_adapter = visual_adapter

    def text_forward(self, cls_names):
        temp = CUSTOM_TEMPLATES[self.args["dataset"]]
        texts = [temp.format(c.replace("_", " ")) for c in cls_names]
        texts = self.tokenizer(texts)
        text_features = self.text_encoder(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.register_buffer('text_features', text_features)

    def _forward_visual_adapter(self, x, task_id):

        _, end_cls = self.get_cls_per_task(task_id)
        text_features = self.text_features[:end_cls, :]

        image_features = self.visual_adapter(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T

        return logits

    def forward(self, image, zeroshot=False, fc_only=False, joint_train=False, train=False, task_id=-1):

        if zeroshot:
            image_features_zs = self.image_encoder(image)
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
            return image_features_zs

        if fc_only:
            return self._forward_visual_adapter(image, task_id)
            # images are stored statistical features

        if joint_train:
            image_features_zs = self.image_encoder(image)
            return self._forward_visual_adapter(image_features_zs, task_id)

        if train:
            image_features_zs = self.image_encoder(image)

            start_cls, end_cls = self.get_cls_per_task(task_id)
            text_features = self.text_features[start_cls:end_cls, :]

            image_features = self.visual_adapter(image_features_zs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T

            return logits
        else:
            image_features_zs = self.image_encoder(image)
            return self._forward_visual_adapter(image_features_zs, task_id)

    def get_cls_per_task(self, task_id):

        if task_id == 0:
            start_cls = 0
            end_cls = self.args["init_cls"]
        else:
            start_cls = self.args["init_cls"] + (task_id - 1) * self.args["increment"]
            end_cls = start_cls + self.args["increment"]

        return start_cls, end_cls