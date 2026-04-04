import logging
import torch
import numpy as np
from models.base import BaseLearner
from utils.inc_net import zsCLIP
from torch.utils.data import DataLoader


num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = zsCLIP(args)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_cls = args["init_cls"]
        self.inc = args['increment']

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):

        self._total_classes = self.args["nb_classes"]

        logging.info("Learning on {}-{}".format(0, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(0, self._total_classes),
                                                      source="train", mode="train", )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self._network.to(self._device)
        # if len(self._multiple_gpus) > 1:
        #     print('Multiple GPUs')
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        all_class_features = self._train()
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module
        return all_class_features


    def _train(self):
        all_class_features = self.feature_record(model=self._network.backbone)
        return all_class_features

    def _eval_cnn(self, loader):
        return

    def feature_record(self, model):

        model.eval()

        all_class_features = {}

        for class_idx in range(0, self._total_classes):
            idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size * 4, shuffle=False, num_workers=4
            )

            vectors = []
            for _, _inputs, _targets in idx_loader:
                with torch.no_grad():
                    _vectors = model(_inputs.to(self._device), zeroshot=True)
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            features_per_cls = vectors

            all_class_features[class_idx] = features_per_cls

        return all_class_features


