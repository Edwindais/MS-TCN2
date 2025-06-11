#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import sys
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)
parser.add_argument('--resume_model_path', type=str, default=None)

parser.add_argument('--num_f_maps', default='64', type=int)

parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)

args = parser.parse_args()
import re

def find_latest_epoch(model_dir):
    """自动查找当前 split 下最大的 epoch 编号"""
    if not os.path.exists(model_dir):
        return None
    epoch_files = [f for f in os.listdir(model_dir) if re.match(r'epoch-(\d+)\.model$', f)]
    if not epoch_files:
        return None
    return max([int(re.findall(r'\d+', f)[0]) for f in epoch_files])
num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_eval = "./data/"+args.dataset+"/splits/val.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.bundle"
features_path = "./data/"+args.dataset+"/features_chopred_15fps/"
gt_path = "./data/"+args.dataset+"/groundtruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

model_dir = "./models/"+args.dataset+"/split_"+args.split
results_dir = "./results/"+args.dataset+"/split_"+args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_eval)

    trainer.train(
        save_dir=model_dir,
        batch_gen=batch_gen,
        val_batch_gen=batch_gen_tst,  # 新增传入
        num_epochs=num_epochs,
        batch_size=bz,
        learning_rate=lr,
        device=device
    )


if args.action == "predict":
    # determine which splits to evaluate
    if args.split == '0':
        splits_to_eval = []
        base_model_dir = os.path.join("models", args.dataset)
        # Collect all valid split directories (skip any 'split_0' placeholder)
        for name in sorted(os.listdir(base_model_dir)):
            m = re.match(r"split_(\d+)$", name)
            if m:
                split_num = m.group(1)
                if split_num == '0':
                    continue
                splits_to_eval.append(split_num)
        # Filter out any splits that have no saved model files
        valid_splits = []
        for split in splits_to_eval:
            model_dir_split = os.path.join("models", args.dataset, f"split_{split}")
            if find_latest_epoch(model_dir_split) is not None:
                valid_splits.append(split)
            else:
                print(f"[Warning] Skipping split {split}: no epoch-*.model files found in {model_dir_split}")
        splits_to_eval = valid_splits
    else:
        splits_to_eval = [args.split]

    all_metrics = {}

    vid_list_file_tst = "./data/"+args.dataset+"/splits/test.bundle"
    for split in splits_to_eval:
        print(f"\n➡️ Predicting Split {split}...")
        model_dir_split = os.path.join("models", args.dataset, f"split_{split}")
        results_dir_split = os.path.join("results", args.dataset, f"split_{split}")

        # select epoch
        if args.resume_model_path:
            predict_epoch = int(args.resume_model_path.split('-')[-1].split('.')[0])
        else:
            predict_epoch = find_latest_epoch(model_dir_split)
            if predict_epoch is None:
                raise ValueError(f"No epoch-*.model files found in {model_dir_split}")
        print(f"✅ Auto-selected epoch {predict_epoch} for split {split}")

        # run prediction
        batch_gen_eval = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen_eval.read_data(vid_list_file_tst)
        metrics = trainer.predict(
            model_dir_split,
            results_dir_split,
            features_path,
            batch_gen_eval,
            predict_epoch,
            actions_dict,
            device,
            sample_rate
        )
        all_metrics[split] = metrics

    # write combined metrics once
    output_file = os.path.join("results", args.dataset, "metrics_all_splits.txt")
    with open(output_file, "w") as f:
        for split, metrics in all_metrics.items():
            f.write(f"Split {split} Metrics:\n")
            for key, val in metrics.items():
                f.write(f"{key}: {val}\n")
            f.write("\n")
    print(f"[Save] Combined metrics for all splits saved to {output_file}")
    sys.exit(0)
