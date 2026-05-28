import logging
import torch
import numpy as np
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.inc_net import MultiProxyCLIP
from torch.utils.data import DataLoader
import time


num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MultiProxyCLIP(args)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_cls = args["init_cls"]
        self.inc = args['increment']
        self.para = args['para']

        self.cls_mean = dict()
        self.cls_cov = dict()

        self.gmms = {}
        self.best_ks = {}

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source="train", mode="train", )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test",)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self._network.to(self._device)

        self._train()

    def _train(self):
        self.gmm_fit(model=self._network.backbone)
        logging.info("best_ks: {}".format(self.best_ks))
        logging.info(
            f"lambda={self.para:.4f}"
        )
        self.visual_adapter_train()

    def visual_adapter_train(self):

        sampled_data_list = []
        for class_id in np.arange(self._known_classes, self._total_classes):
            sampled_data = self.cls_mean[class_id].unsqueeze(0)
            sampled_data_list.append(sampled_data)

        sampled_data_list = torch.cat(sampled_data_list, dim=0).float().to(self._device)

        self._network.update_fc(nb_classes=self._total_classes, sampling_features=sampled_data_list, nb_proxy=1)

    def _eval_cnn(self, loader):

        calc_task_acc = False

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        self._network.eval()

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs, image_features = self._network(inputs, zeroshot=True)

            features_np = image_features.cpu().numpy()
            log_probs = np.zeros((features_np.shape[0], self._total_classes))

            for class_idx in range(self._total_classes):

                gmm = self.gmms[class_idx]
                class_log_probs = gmm.score_samples(features_np)
                log_probs[:, class_idx] = class_log_probs

            log_probs_tensor = torch.from_numpy(log_probs).float().to(self._device)

            logits_gmm = torch.softmax(log_probs_tensor, dim=-1)

            outputs += self.para * logits_gmm

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            if calc_task_acc:
                task_ids = (targets - self.args["init_cls"]) // self.args["increment"] + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    start_cls, end_cls = self. get_cls_per_task(task_id)
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]

                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()

                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))

        return np.concatenate(y_pred), np.concatenate(y_true)

    def get_cls_per_task(self, task_id):

        if task_id == 0:
            start_cls = 0
            end_cls = self.args["init_cls"]
        else:
            start_cls = self.args["init_cls"] + (task_id - 1) * self.args["increment"]
            end_cls = start_cls + self.args["increment"]

        return start_cls, end_cls

    def gmm_fit(self, model):
        from sklearn.mixture import GaussianMixture

        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
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
            self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
            self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)

            X = features_per_cls.cpu().numpy()

            n_components_list = [1, 2, 3, 4, 5]
            # n_components_list = [1]
            bic_scores = []
            gmm_models = []

            for k in n_components_list:
                gmm = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-4, random_state=self.args["seed"])
                gmm.fit(X)
                bic = gmm.bic(X)
                bic_scores.append(bic)
                gmm_models.append(gmm)

            valid_indices = [i for i, bic in enumerate(bic_scores)]
            best_idx = valid_indices[np.argmin([bic_scores[i] for i in valid_indices])]
            best_k = n_components_list[best_idx]
            best_gmm = gmm_models[best_idx]

            self.best_ks[class_idx] = best_k
            self.gmms[class_idx] = best_gmm

