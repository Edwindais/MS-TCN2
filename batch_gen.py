#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
    # Read video ID list, one per line, without dropping the last entry
    # and skip any empty lines.
        with open(vid_list_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.list_of_examples = lines
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        # 1) 取出本次 batch 的视频 ID 列表
        batch = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []

        # 2) 遍历每个视频，加载特征和标签
        for vid in batch:
            feat_path = os.path.join(self.features_path, vid.split('.')[0] + '.pth')
            features = torch.load(feat_path)  # Tensor (C, T)

            gt_path = os.path.join(self.gt_path, vid + '.txt')
            with open(gt_path, 'r') as f:
                content = [line.strip() for line in f if line.strip()]

            # 3) 构建 frame-level 类别索引数组
            num_frames = min(features.size(1), len(content))
            classes = np.zeros(num_frames, dtype=np.int64)
            unknown = set()
            for i in range(num_frames):
                label = content[i]
                if label in self.actions_dict:
                    classes[i] = self.actions_dict[label]
                else:
                    unknown.add(label)
            if unknown:
                raise KeyError(f"Unknown actions: {unknown}")

            # 4) 下采样
            feat_ds = features[:, ::self.sample_rate]          # (C, T')
            tgt_ds  = torch.from_numpy(classes[::self.sample_rate])  # (T',)

            batch_input.append(feat_ds)
            batch_target.append(tgt_ds)

        # 5) 计算对齐后各样本长度（裁剪到 feat 和 label 的较短者）
        true_lengths = []
        for feat, tgt in zip(batch_input, batch_target):
            true_lengths.append(min(feat.size(1), tgt.size(0)))
        L = max(true_lengths)           # 本 batch 统一长度

        # 6) 预分配张量
        B = len(batch_input)
        C = batch_input[0].size(0)
        batch_input_tensor  = torch.zeros(B, C, L, dtype=torch.float)
        batch_target_tensor = torch.ones(B, L, dtype=torch.long) * -100
        mask                = torch.zeros(B, self.num_classes, L, dtype=torch.float)

        # 7) 填充（先裁剪到各自最短 true_len）
        for i in range(B):
            feat = batch_input[i]
            tgt  = batch_target[i]
            Ti   = feat.size(1)
            Tj   = tgt.size(0)
            Li   = min(Ti, Tj)          # 统一裁剪长度

            batch_input_tensor[i, :, :Li] = feat[:, :Li]
            batch_target_tensor[i, :Li]   = tgt[:Li]
            mask[i, :, :Li]               = 1.0

        # 8) 返回四元组：输入、标签、mask、以及原始 vid 列表
        return batch_input_tensor, batch_target_tensor, mask, batch
