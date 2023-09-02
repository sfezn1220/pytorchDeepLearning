""" 训练、测试的主函数； """

import os
import torch
import argparse
import copy
import yaml
import logging


def get_args():
    parser = argparse.ArgumentParser(description='training models.')

    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='0 for gpu and -1 for cpu')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # GPU or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if str(args.gpu) == "-1":
        logging.info(f"Use device: CPU.")
    elif str(args.gpu) == "0":
        logging.info(f"Use device: GPU {str(args.gpu)}.")
    else:
        raise ValueError(f"\"--gpu\" must in [-1, 0], while input is {str(args.gpu)}")

    # Set random seed
    torch.manual_seed(777)

    # symbol_table  # TODO

    # 读取 configs.yaml
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['shuffle'] = False

    # 数据集：
    train_dataset = Dataset(args.data_type, args.train_data, symbol_table,
                            train_conf, args.bpe_model, non_lang_syms, True)
    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         symbol_table,
                         cv_conf,
                         args.bpe_model,
                         non_lang_syms,
                         partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)