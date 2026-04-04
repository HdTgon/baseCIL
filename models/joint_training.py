import logging
import torch
import numpy as np
from tqdm import tqdm
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.inc_net import CustomCLIP
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import optim


num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = CustomCLIP(args)

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

        logging.info("Learning on {}-{}".format(0, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="train", mode="train",)
        # joint training
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

        network_params = [p for name, p in self._network.backbone.named_parameters() if
                          "visual_adapter" in name and p.requires_grad]
        optimizer = optim.SGD(
            network_params,
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"],
                                                         eta_min=self.args["min_lr"])

        self.visual_adapter_train(train_loader, optimizer, scheduler)

    def visual_adapter_train(self, train_loader, optimizer, scheduler):

        epochs = self.args["epochs"]
        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs, zeroshot=False, fc_only=False, joint_train=True, train=False, task_id=self._cur_task)

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, predicted = torch.max(logits, dim=1)

                correct += predicted.eq(targets).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f} ".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
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
                outputs = self._network(inputs, zeroshot=False, fc_only=False, joint_train=False, train=False, task_id=self._cur_task)

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            if calc_task_acc:
                task_ids = (targets - self.args["init_cls"]) // self.args["increment"] + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    start_cls, end_cls = self.get_cls_per_task(task_id)
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
