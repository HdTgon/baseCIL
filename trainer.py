import logging
import torch
import copy
import os
import sys
from utils.data_manager import DataManager
from utils import factory
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    if args["model_name"] in ['joint_training', 'partdata_statistics_cov', 'alldata_statistics_imbalance', 'multi_proxy', 'mahala_distance', 'gmm', 'visual_classifier', 'tsne', 'adaptive_lambda', 'gmm_ensemble']:
        log_name = "logs/CLIP_CIL/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args["increment"])
    elif args["model_name"] == 'ensemble':
        log_name = "logs/CLIP_CIL/{}_{}/{}/{}/{}".format(args["model_name"], args["para"], args["dataset"], init_cls, args["increment"])
    else:
        log_name = "logs/CLIP_CIL/{}_{}/{}/{}/{}".format(args["model_name"], args["statistics_method"], args["dataset"], init_cls, args["increment"])

    if not os.path.exists(log_name):
        os.makedirs(log_name)

    if args["model_name"] in ['joint_training', 'alldata_statistics_imbalance', 'mahala_distance', 'gmm', 'visual_classifier', 'tsne', 'gmm_ensemble']:
        logfilename = "logs/CLIP_CIL/{}/{}/{}/{}/{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["seed"],
            args["backbone_name"],
        )
    elif args["model_name"] == 'partdata_statistics_cov':
        logfilename = "logs/CLIP_CIL/{}/{}/{}/{}/{}few_shot_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["few_shot"],
            args["seed"],
            args["backbone_name"],
        )
    elif args["model_name"] == 'multi_proxy':
        logfilename = "logs/CLIP_CIL/{}/{}/{}/{}/{}_proxy_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["nb_proxy"],
            args["seed"],
            args["backbone_name"],
        )
    elif args["model_name"] == 'ensemble':
        logfilename = "logs/CLIP_CIL/{}_{}/{}/{}/{}/{}_{}_{}".format(
            args["model_name"],
            args["para"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["seed"],
            args["backbone_name"],
        )
    elif args["model_name"] == 'adaptive_lambda':
        logfilename = "logs/CLIP_CIL/{}/{}/{}/{}/temp{}_gating{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["temp"],
            args["gating"],
            args["seed"],
            args["backbone_name"],
        )
    elif args["model_name"] == 'gmm_ensemble':
        logfilename = "logs/CLIP_CIL/{}/{}/{}/{}/para{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["para"],
            args["seed"],
            args["backbone_name"],
        )
    else:
        logfilename = "logs/CLIP_CIL/{}_{}/{}/{}/{}/{}shot_{}_{}".format(
            args["model_name"],
            args["statistics_method"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["num_sampled"],
            args["seed"],
            args["backbone_name"],
        )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout)
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args
    )

    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks
    args["shuffled_cls_names"] = data_manager.class_names
    args["file_name"] = log_name
    model = factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):

        if args["model_name"] == 'tsne':
            all_class_features = model.incremental_train(data_manager)

            save_path = "{}/class_features_{}.pth".format(log_name, args["dataset"])
            torch.save(all_class_features, save_path)
            break

        else:
            logging.info("All params: {}".format(count_parameters(model._network)))
            logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
            model.incremental_train(data_manager)
            cnn_accy = model.eval_task()
            model.after_task()

            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))

    del model
    torch.cuda.empty_cache()

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
