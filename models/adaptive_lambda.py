import logging
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from models.base import BaseLearner
from utils.inc_net import MultiProxyCLIP
from torch.utils.data import DataLoader

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MultiProxyCLIP(args)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]
        self.para = 0.0
        self.lambda_0 = 1
        # self.temp = 0.3
        self.temp = args["temp"]
        self.gating = args["gating"]
        self.s = 0.0

        # ===== Class statistics =====
        self.cls_mean = dict()
        self.cls_cov_inv = dict()

    def after_task(self):

        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        self.data_manager = data_manager

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train"
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test"
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self._network.to(self._device)
        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):

        self._compute_mean(self._network.backbone)
        self.update_adaptive_lambda_global()
        self.visual_adapter_train()

    def visual_adapter_train(self):

        self._network.freeze()

        sampled_data_list = []
        for class_id in range(self._known_classes, self._total_classes):
            sampled_data_list.append(self.cls_mean[class_id].unsqueeze(0))

        sampled_data = torch.cat(sampled_data_list, dim=0).float().to(self._device)
        self._network.update_fc(
            nb_classes=self._total_classes,
            sampling_features=sampled_data,
            nb_proxy=1
        )

    def update_adaptive_lambda_global(self):

        gating = self.gating

        if self.s <= 0.0:
            self.para = 0
        else:
            self.para = self.lambda_0 * min(max(self.s - gating, 0)/self.temp, 1)

        logging.info(
            f"[Task {self._cur_task}] gating={gating:.2f}, temp={self.temp:.3f}, Silhouette={self.s:.4f}, lambda={self.para:.4f}"
        )

    def mahala_distance(self, image_feature, mean, cov_inv):

        diff = image_feature - mean
        temp = torch.matmul(diff, cov_inv)
        dist_sq = torch.sum(temp * diff, dim=1)
        return torch.sqrt(dist_sq + 1e-8)

    def distance_to_score(self, distance):

        min_d, _ = distance.min(dim=1, keepdim=True)
        max_d, _ = distance.max(dim=1, keepdim=True)
        range_d = torch.clamp(max_d - min_d, min=1e-8)
        normalized = (distance - min_d) / range_d
        return 1.0 - normalized

    def _eval_cnn(self, loader):

        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs, image_features = self._network(inputs, zeroshot=True)

            mahala_dist = torch.zeros_like(outputs)

            for class_idx in range(self._total_classes):
                mean = self.cls_mean[class_idx]
                cov_inv = self.cls_cov_inv[class_idx]
                dist = self.mahala_distance(image_features, mean, cov_inv)
                mahala_dist[:, class_idx] = dist

            score_mahala = self.distance_to_score(mahala_dist)
            outputs = outputs + self.para * score_mahala

            preds = torch.topk(outputs, k=self.topk, dim=1)[1]
            y_pred.append(preds.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_mean(self, model):
        model.eval()

        current_feats = []
        current_labels = []

        # ===== Compute statistics for new classes =====
        for class_idx in range(self._known_classes, self._total_classes):
            dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test"
            )
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size * 4,
                shuffle=False,
                num_workers=4
            )

            feats = []
            for _, inputs, _ in loader:
                with torch.no_grad():
                    feat = model(inputs.to(self._device), zeroshot=True)
                feats.append(feat)

            feats = torch.cat(feats, dim=0)

            # class statistics
            mean = feats.mean(dim=0)
            cov = torch.cov(feats.T) + torch.eye(feats.shape[1], device=feats.device) * 1e-4

            self.cls_mean[class_idx] = mean
            self.cls_cov_inv[class_idx] = torch.linalg.inv(cov)

            # collect features for silhouette
            current_feats.append(feats.cpu().numpy())
            current_labels.extend([class_idx] * len(feats))

        # ===== Compute silhouette for current task =====
        SIL_GROUP = 5
        cur_s = None

        feats_all = np.vstack(current_feats)
        labels_all = np.array(current_labels)

        unique_classes = np.unique(labels_all)
        num_classes = len(unique_classes)

        num_groups = num_classes // SIL_GROUP

        sil_scores = []

        if num_groups > 0:
            for i in range(num_groups):
                group_classes = unique_classes[i * SIL_GROUP:(i + 1) * SIL_GROUP]
                mask = np.isin(labels_all, group_classes)

                feats_g = feats_all[mask]
                labels_g = labels_all[mask]

                if len(np.unique(labels_g)) >= 2:
                    try:
                        s_g = silhouette_score(feats_g, labels_g)
                        sil_scores.append(s_g)
                    except ValueError:
                        pass

            if len(sil_scores) > 0:
                cur_s = float(np.mean(sil_scores))

        # ===== Update accumulated silhouette =====
        if cur_s is not None:
            prev_classes = self._known_classes
            cur_classes = num_groups * SIL_GROUP
            total_classes = prev_classes + cur_classes

            self.s = (self.s * prev_classes + cur_s * cur_classes) / total_classes





