from mid import MID
import argparse
import os
import yaml
from easydict import EasyDict
import numpy as np
import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='')
    return parser.parse_args()

def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset[:]
    #pdb.set_trace()
    config = EasyDict(config)

    # この時点でモデル自体はダウンロードされる
    agent = MID(config)

    sampling="ddim"
    steps=5

    if config["eval_mode"]:
        agent.eval()
    else:
        agent.train()

if __name__ == '__main__':
    main()
