from torch import nn
from backbone import cusclip, vcclip
from backbone.linears import CosineLinear
import copy


class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        print('This is for the BaseNet initialization.')

        name = args["model_name"].lower()
        if name == "visual_classifier":
            self.backbone = vcclip.CLIP(args)
        else:
            self.backbone = cusclip.myCLIP(args)

        self._device = args["device"][0]


class CustomCLIP(BaseNet):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self.out_dim = self.backbone.out_dim

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x, zeroshot=False, fc_only=False, joint_train=False, train=False, task_id=-1):

        return self.backbone.forward(x, zeroshot, fc_only, joint_train, train, task_id)


class MultiProxyCLIP(BaseNet):
    def __init__(self, args):
        super().__init__(args)
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self.out_dim = self.backbone.out_dim
        self.fc = None

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def generate_fc(self, in_dim, out_dim, nb_proxy):
        fc = CosineLinear(in_dim, out_dim, nb_proxy=nb_proxy, to_reduce=True)
        return fc

    def update_fc(self, nb_classes, sampling_features, nb_proxy):

        fc = self.generate_fc(self.out_dim, nb_classes, nb_proxy).to(self._device)
        fc.set_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            old_weight = copy.deepcopy(self.fc.weight.data)
            fc.weight.data[: old_nb_classes,] = nn.Parameter(old_weight)
            fc.weight.data[old_nb_classes:,] = sampling_features
            # (nb_classes*nb_proxy, 512)
        else:
            fc.weight.data = sampling_features
            # 10*nb_proxy,512

        del self.fc
        self.fc = fc

    def forward(self, x, zeroshot=True):

        image_features = self.backbone.forward(x, zeroshot=zeroshot, fc_only=False, joint_train=False, train=False, task_id=-1)

        logits = self.fc(image_features)

        return logits, image_features


class zsCLIP(BaseNet):
    def __init__(self, args):
        super().__init__(args)
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self.out_dim = self.backbone.out_dim
        self.fc = None

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x, zeroshot=True):

        image_features = self.backbone.forward(x, zeroshot=zeroshot, fc_only=False, joint_train=False, train=False, task_id=-1)

        return image_features


class vclassCLIP(BaseNet):
    def __init__(self, args):
        super().__init__(args)
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self.out_dim = self.backbone.out_dim
        self.fc = None

    def generate_fc(self, in_dim, out_dim):
        fc = nn.Linear(in_dim, out_dim)
        fc.requires_grad_(True)
        return fc

    def update_fc(self, nb_classes):

        fc = self.generate_fc(self.out_dim, nb_classes).to(self._device)

        del self.fc
        self.fc = fc

    def forward(self, image, train=False):

        if train:
            logits = self.fc(image)
            return logits

        else:
            image_features = self.backbone.forward(image)
            logits = self.fc(image_features)

            return logits, image_features
