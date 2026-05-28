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
    parser.add_argument('--config', type=str, default='./exps/gmm_ensemble/ucf.json')
    parser.add_argument('--para', type=float, default=0.1)
    return parser


if __name__ == '__main__':
    main()

