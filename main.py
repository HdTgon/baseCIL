import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)

    train(args)


def load_json(setting_path):
    with open(setting_path) as json_file:
        param = json.load(json_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='test_statistics.')
    parser.add_argument('--config', type=str, default='./exps/gmm_ensemble/cars.json')
    parser.add_argument('--num_sampled', type=int, default=256)
    parser.add_argument('--few_shot', type=int, default=0)
    parser.add_argument('--nb_proxy', type=int, default=0)
    parser.add_argument('--para', type=float, default=1)
    parser.add_argument('--temp', type=float, default=0.3)
    parser.add_argument('--gating', type=float, default=0.05)
    return parser


if __name__ == '__main__':
    main()

