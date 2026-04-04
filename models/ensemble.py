import logging
import torch
import numpy as np
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.inc_net import MultiProxyCLIP
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MultiProxyCLIP(args)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_cls = args["init_cls"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args['increment']
        self.para = args['para']

        self.cls_mean = dict()
        self.cls_cov = dict()

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
        # if len(self._multiple_gpus) > 1:
        #     print('Multiple GPUs')
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

    def _train(self, train_loader, test_loader):

        self._compute_mean(self._network.backbone)
        self.visual_adapter_train()

    def visual_adapter_train(self):

        self._network.freeze()

        sampled_data_list = []
        for class_id in np.arange(self._known_classes, self._total_classes):
            sampled_data = self.cls_mean[class_id].unsqueeze(0)
            sampled_data_list.append(sampled_data)

        sampled_data_list = torch.cat(sampled_data_list, dim=0).float().to(self._device)

        self._network.update_fc(nb_classes=self._total_classes, sampling_features=sampled_data_list, nb_proxy=1)

    def mahala_distance(self, image_feature, mean, cov):

        diff = image_feature - mean
        inv_cov = torch.inverse(cov)
        temp = torch.matmul(diff, inv_cov)
        distances_squared = torch.sum(temp * diff, dim=1)

        return torch.sqrt(distances_squared)

    def distance_to_score(self, distance):

        min_d, _ = distance.min(dim=1, keepdim=True)
        max_d, _ = distance.max(dim=1, keepdim=True)
        range_d = max_d - min_d

        range_d = torch.where(range_d == 0, torch.ones_like(range_d) * 1e-8, range_d)

        normalized = (distance - min_d) / range_d
        score = 1.0 - normalized

        return score

    def _eval_cnn(self, loader):
        calc_task_acc = True

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs, image_features = self._network(inputs, zeroshot=True)

            mahala_distance = torch.zeros_like(outputs)

            for class_idx in range(0, self._total_classes):
                mean = self.cls_mean[class_idx]
                cov = self.cls_cov[class_idx]
                distance = self.mahala_distance(image_features, mean, cov)
                mahala_distance[:, class_idx] = distance

            score_mahala = self.distance_to_score(mahala_distance)

            outputs += self.para*score_mahala

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

    def _compute_mean(self, model):

        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            if self.args["model_name"] == "partdata_statistics_cov":
                idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                              source="train", mode="test",
                                                              few_shot=self.args["few_shot"])
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.args["few_shot"], shuffle=False, num_workers=4
                )
            else:
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

    def sampling(self, class_id, num_sampled):

        mean = self.cls_mean[class_id].to(self._device)
        cov = self.cls_cov[class_id].to(self._device)

        m = MultivariateNormal(mean.float(), cov.float())
        sampled_data = m.sample(sample_shape=(num_sampled,))
        sampled_label = [class_id] * num_sampled

        return sampled_data, sampled_label

