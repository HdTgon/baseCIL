import logging
import torch
import numpy as np
from tqdm import tqdm
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.inc_net import vclassCLIP
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import optim


num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = vclassCLIP(args)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_cls = args["init_cls"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args['increment']

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
        self._network.update_fc(self._total_classes)
        self._train(self.train_loader, self.test_loader)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

    def _train(self, train_loader, test_loader):

        network_params = [p for name, p in self._network.named_parameters() if p.requires_grad]

        optimizer = optim.SGD(
            network_params,
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"],
                                                         eta_min=self.args["min_lr"])

        self._compute_mean(self._network)
        self.visual_adapter_train(train_loader, optimizer, scheduler)

    def visual_adapter_train(self, train_loader, optimizer, scheduler):

        epochs = self.args["epochs"]
        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            num_sampled = self.args["num_sampled"]

            sampled_data_list, sampled_labels_list = [], []
            for class_id in range(len(self.cls_mean)):
                sampled_data, sampled_label = self.sampling(class_id, num_sampled)
                sampled_data_list.append(sampled_data)
                sampled_labels_list.extend(sampled_label)

            sampled_data_list = torch.cat(sampled_data_list, dim=0).float().to(self._device)
            sampled_labels_list = torch.tensor(sampled_labels_list).long().to(self._device)

            total_samples = sampled_data_list.shape[0]

            inputs = sampled_data_list
            targets = sampled_labels_list

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0

            for _iter in range(len(self.cls_mean)):
                inp = inputs[_iter * num_sampled:(_iter + 1) * num_sampled]
                tgt = targets[_iter * num_sampled:(_iter + 1) * num_sampled]

                logits = self._network(inp, train=True)
                loss = F.cross_entropy(logits, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, predicted = torch.max(logits, dim=1)

                correct += predicted.eq(tgt.expand_as(predicted)).cpu().sum()
                total += len(tgt)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f} ".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / total_samples,
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        calc_task_acc = True

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs, _ = self._network(inputs)

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
                    _, _vectors = model(_inputs.to(self._device))
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            if self.args["statistics_method"] == 'covariance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            elif self.args["statistics_method"] == 'variance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device))
            elif self.args["statistics_method"] == 'prototype':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)

    def sampling(self, class_id, num_sampled):

        if self.args["statistics_method"] in ['covariance', 'variance']:
            mean = self.cls_mean[class_id].to(self._device)
            cov = self.cls_cov[class_id].to(self._device)
            if self.args["statistics_method"] == 'variance':
                cov = torch.diag(cov)
            m = MultivariateNormal(mean.float(), cov.float())
            sampled_data = m.sample(sample_shape=(num_sampled,))
            sampled_label = [class_id] * num_sampled

            return sampled_data, sampled_label

        elif self.args["statistics_method"] == "prototype":
            mean = self.cls_mean[class_id].to(self._device)
            expand_mean = mean.unsqueeze(0).expand(num_sampled, -1)
            noisy_prototype = expand_mean + torch.randn_like(expand_mean) * self.args["sample_noise"]
            sampled_label = [class_id] * num_sampled

            return noisy_prototype, sampled_label
