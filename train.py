import os
import shutil
from argparse import ArgumentParser
from main import ACAL
from addict import Dict
import yaml
import datetime

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str)
    parser.add_argument("--pre_batch_size", type=int, default=0)
    parser.add_argument("--pre_lr", type=float, default=0)
    parser.add_argument("--optimizer", type=str, default=None)
    
    return parser.parse_args()

def main():
    argparse = get_args()
    args = Dict(yaml.safe_load(open(argparse.cfg)))
    dt_now = datetime.datetime.now()
    #fetch current time and naming the result path after it
    log_name = str(dt_now.year) + "_" + str(dt_now.month) + "_" + str(dt_now.day) + "_" + str(dt_now.hour) + "_" + str(dt_now.minute) + "_" + str(dt_now.second)
    result_path = os.path.join("result", log_name)
    os.mkdir(result_path)
    shutil.copy(argparse.cfg, result_path)
    args.result_path = result_path
    args.log_name = log_name
    if argparse.pre_batch_size > 0:
        args.pre_batch_size = argparse.pre_batch_size
    if argparse.pre_lr > 0:
        args.pre_lr = argparse.pre_lr
    if argparse.optimizer:
        args.optimizer = argparse.optimizer

    if args.pretrain:
        model = ACAL(args)
        model.pretrain(args)
    model = ACAL(args)
    if args.mode == "reconstruct":
        model.train_recon(args)
    elif args.mode == "classify":
        model.train(args)
    elif args.mode == "finetune":
        model.pretrain(args, ft=True)
if __name__ == '__main__':
    main()

