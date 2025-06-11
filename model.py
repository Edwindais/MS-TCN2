#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
import glob
import os
import utils
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import matplotlib
import numpy as np
from matplotlib import font_manager
# ===== Edit Distance Metric =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out

def cleanup_old_models(model_dir, keep_last_n=5):
        """
        保留最近 keep_last_n 个模型 + best_model.pth，其余全部删除。
        """
        model_files = sorted(
            glob.glob(os.path.join(model_dir, "epoch-*.pth")),
            key=os.path.getmtime
        )
        
        # 不删除 best_model
        best_model_path = os.path.join(model_dir, "best_model.pth")

        if len(model_files) > keep_last_n:
            old_files = model_files[:-keep_last_n]
            for f in old_files:
                if os.path.abspath(f) != os.path.abspath(best_model_path):
                    try:
                        os.remove(f)
                        print(f"[Cleanup] Deleted old model: {f}")
                    except Exception as e:
                        print(f"[Warning] Could not delete {f}: {e}")
class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        class_weights = torch.tensor([0.0896, 0.0968, 0.0985, 0.0961, 0.0901, 0.0895, 0.0984, 0.0954, 0.1188,
        0.1081, 0.1081, 0.1024], dtype=torch.float).to(device) 
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")   
    
    def train(self, save_dir, batch_gen, val_batch_gen, num_epochs, batch_size, learning_rate, device, resume=False):
        from torch.utils.tensorboard import SummaryWriter
        from tqdm import tqdm

        self.model.to(device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        start_epoch = 0

        writer = SummaryWriter(log_dir=save_dir + '/tensorboard_logs')

        # Early stopping variables
        best_val_bal_acc = 0
        patience = 3  # You can adjust the patience
        patience_counter = 0

        if resume:
            checkpoint = torch.load(save_dir + "/checkpoint.pth")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            batch_gen.reset()
            pbar = tqdm(total=len(batch_gen.list_of_examples), desc=f"Epoch {epoch+1}/{num_epochs}")
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    mse_loss = 0.15 * torch.mean(
                        torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:]
                    )
                    loss += ce_loss + mse_loss
                # Boundary loss (assumed to be implemented)
                # Adding boundary loss example, replace with actual boundary loss if available
                if hasattr(self, 'boundary_loss'):
                    b_loss = self.boundary_loss(predictions[-1], batch_target, mask)
                    loss += b_loss

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                pbar.update(batch_size)
            pbar.close()

            scheduler.step()

            train_acc = float(correct) / total if total > 0 else 0
            avg_loss = epoch_loss / len(batch_gen.list_of_examples)
            logger.info(f"[epoch {epoch + 1}]: epoch loss = {avg_loss:.6f},   acc = {train_acc:.6f}")
            writer.add_scalar('Loss/train', avg_loss, epoch + 1)
            writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch + 1)

            # Validation and early stopping every 10 epochs
            if (epoch + 1) % 20 == 0:
                val_loss, val_acc, val_bal_acc = self.test(val_batch_gen, batch_size, device)
                logger.info(f"[epoch {epoch + 1}]: val loss = {val_loss:.6f},   val acc = {val_acc:.6f},   val balanced acc = {val_bal_acc:.6f}")
                writer.add_scalar('Loss/val', val_loss, epoch + 1)
                writer.add_scalar('Accuracy/val', val_acc, epoch + 1)
                writer.add_scalar('BalancedAccuracy/val', val_bal_acc, epoch + 1)
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
                # Early stopping logic only triggered when validation is run
                if val_bal_acc > best_val_bal_acc:
                    best_val_bal_acc = val_bal_acc
                    patience_counter = 0
                    torch.save(self.model.state_dict(), save_dir + "/best_model.pth")
                else:
                    patience_counter += 1
                    logger.info(f"EarlyStopping counter: {patience_counter} out of {patience}")
                    if patience_counter >= patience:
                        logger.info("Early stopping triggered.")
                        writer.close()
                        return  # Exit training early
            cleanup_old_models(save_dir, keep_last_n=5)

            # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            # Save checkpoint for resuming
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_dir + "/checkpoint.pth")
        writer.close()

    def test(self, val_batch_gen, batch_size, device):
        self.model.eval()
        val_batch_gen.reset()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            while val_batch_gen.has_next():
                batch_input, batch_target, mask,batch = val_batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    mse_loss = 0.15 * torch.mean(
                        torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:]
                    )
                    loss += ce_loss + mse_loss
                total_loss += loss.item()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                # 收集有效帧用于 balanced accuracy
                valid = mask[:, 0, :].squeeze(1).bool()
                all_preds.extend(predicted[valid].cpu().numpy())
                all_targets.extend(batch_target[valid].cpu().numpy())

        # 计算 avg_loss 和 accuracy
        avg_loss = total_loss / len(val_batch_gen.list_of_examples)
        accuracy = float(correct) / total if total > 0 else 0
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)

        print("Predicted labels distribution:", np.unique(all_preds, return_counts=True))
        print("True labels distribution:", np.unique(all_targets, return_counts=True))
        self.model.train()
        return avg_loss, accuracy, balanced_acc
    
  
    def segment_bars_with_confidence(self, save_path, confidence, *labels):
        import matplotlib.pyplot as plt
        num_pics = len(labels) + 1
        color_map = plt.get_cmap('seismic')

        axprops = dict(xticks=[], yticks=[], frameon=False)
        barprops = dict(aspect='auto', cmap=color_map,
                        interpolation='nearest', vmin=0)
        fig = plt.figure(figsize=(15, num_pics * 1.5))

        interval = 1 / (num_pics+1)
        for i, label in enumerate(labels):
            i = i + 1
            ax1 = fig.add_axes([0, 1-i*interval, 1, interval])
            ax1.imshow([label], **barprops)

        # Only plot confidence if it's not None
        if confidence is not None:
            ax4 = fig.add_axes([0, interval, 1, interval])
            ax4.set_xlim(0, len(confidence))
            ax4.set_ylim(0, 1)
            ax4.plot(range(len(confidence)), confidence)
            ax4.plot(range(len(confidence)), [0.3] * len(confidence), color='red', label='0.5')

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def save_horizontal_comparison(self, pred, gt, confidence,
                                name, save_path, actions_dict):

        # ---------- 字体 ----------
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
        FONT = {"fontname": font_name, "fontweight": "bold", "fontsize": 18}

        # ---------- 数据 ----------
        pred       = np.asarray(pred).squeeze().tolist()
        gt         = np.asarray(gt).squeeze().tolist()
        confidence = np.asarray(confidence).squeeze().tolist()

        # ---------- 画布 ----------
        fig, ax = plt.subplots(
            3, 1, figsize=(18, 6), sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 0.5]}
        )
        # 给 legend 预留 20% 宽度
        fig.subplots_adjust(left=0.05, right=0.80, top=0.90, bottom=0.10, hspace=0.25)

        # ---------- 颜色 ----------
        label_colors = {
            0:'#1f77b4', 1:'#aec7e8', 2:'#ffbb78', 3:'#98df8a',
            4:'#ff9896', 5:'#c5b0d5', 6:'#8c564b', 7:'#e377c2',
            8:'#7f7f7f', 9:'#bcbd22', 10:'#17becf', 11:'#9edae5'
        }

        # ---------- Pred / GT 条纹 ----------
        for i, row in enumerate([pred, gt]):
            for t, lab in enumerate(row):
                ax[i].axvline(t, color=label_colors[lab], linewidth=8)
            ax[i].set_yticks([])
            ax[i].set_ylabel("Pred" if i == 0 else "GT", **FONT)
            ax[i].tick_params(axis="x", labelsize=12)

        # ---------- Confidence ----------
        ax[2].plot(confidence, color="blue", linewidth=1)
        ax[2].set_ylabel("Confidence", **FONT)
        ax[2].set_xlabel("Time (frames)", **FONT)
        ax[2].set_ylim(0, 1.0)
        ax[2].tick_params(labelsize=12)

        # ---------- 坐标刻度加粗 ----------
        for axis in ax:
            for lbl in axis.get_xticklabels() + axis.get_yticklabels():
                lbl.set_fontproperties(font_manager.FontProperties(fname=font_path))
                lbl.set_weight("bold")

        # ---------- 标题 ----------
        fig.suptitle(f"MS_TCN2 Prediction vs Ground Truth: {name}", **FONT)

        # ---------- Legend ----------
        legend_handles = [
            mpatches.Patch(color=color, label=lab)
            for lab, idx in actions_dict.items()
            for k, color in label_colors.items() if k == idx
        ]

        # 用 fig.legend ‑‑ 放在 figure 坐标 (0.82,0.5) 处，垂直居中
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.82, 0.65),   # (x,y) in figure fraction
            borderaxespad=0.0,
            ncol=1,
            fontsize=12,
            frameon=True,
            edgecolor="black"
        )

        # ---------- 保存 ----------
        save_file = os.path.join(save_path, f"{name}_comparison.pdf")
        plt.savefig(save_file, format="pdf", bbox_inches="tight")
        plt.close()

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, device, sample_rate):
        import os, time
        import torch.nn.functional as F

        self.model.eval()
        self.model.to(device)
        ckpt_path = os.path.join(model_dir, f"epoch-{epoch}.model")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        batch_gen_tst.reset()
        all_preds, all_gts = [], []
        t0 = time.time()

        while batch_gen_tst.has_next():
            inp, tgt, mask, vids = batch_gen_tst.next_batch(1)
            vid = vids[0]

            inp  = inp.to(device)  # (1, C, T)
            mask = mask.to(device)

            outputs = self.model(inp)  # list of (1, num_classes, T)

            for stage_idx, out in enumerate(outputs):
                # 得到每帧置信度和预测结果
                prob      = F.softmax(out, dim=1)            # (1, K, T)
                conf, pred = torch.max(prob, dim=1)          # (1, T),(1, T)
                conf_np   = conf.detach().squeeze(0).cpu().numpy().tolist()
                pred_np   = pred.detach().squeeze(0).cpu().numpy().astype(int)
                tgt_np    = tgt.squeeze(0).cpu().numpy().astype(int)

                # 5.1) 时序热力图 + confidence
                heat_save = os.path.join(results_dir, f"{vid}_stage{stage_idx}.png")
                self.segment_bars_with_confidence(
                    heat_save,
                    conf_np,
                    tgt_np.tolist(),
                    pred_np.tolist()
                )

                # 5.2) 最后一阶段额外并排对比图
                if stage_idx == len(outputs) - 1:
                    self.save_horizontal_comparison(
                        pred=pred_np,
                        gt=tgt_np,
                        confidence=conf_np,
                        name=vid,
                        save_path=results_dir,
                        actions_dict=actions_dict
                    )
                # Write frame-level predictions to .txt for evaluation
                txt_path = os.path.join(results_dir, f"{vid}.txt")
                with open(txt_path, "w") as f:
                    f.write("### Frame level recognition: ###\n")
                    f.write(" ".join(map(str, pred_np.tolist())))

            all_preds.extend(pred_np.tolist())
            all_gts.extend(tgt_np.tolist())

            acc = accuracy_score(all_gts, all_preds)
            f1  = f1_score(all_gts, all_preds, average='macro')
            bal = balanced_accuracy_score(all_gts, all_preds)
            # mAP
            K = len(actions_dict)
            gt_oh   = np.eye(K)[all_gts]
            pred_oh = np.eye(K)[all_preds]
            mAP = average_precision_score(gt_oh, pred_oh, average='macro')

            # ===== 新增：分段级 F1 @ IoU 阈值 10%,25%,50% =====
            def extract_segments(labels):
                """
                将一维标签序列合并成 [(label, start_frame, end_frame), …]
                """
                segments = []
                start = 0
                curr = labels[0]
                for i, l in enumerate(labels[1:], 1):
                    if l != curr:
                        segments.append((curr, start, i-1))
                        curr = l
                        start = i
                segments.append((curr, start, len(labels)-1))
                return segments

            def segment_iou(seg1, seg2):
                """计算两个段 (s1,e1) 与 (s2,e2) 的 IoU"""
                s1, e1 = seg1
                s2, e2 = seg2
                inter = max(0, min(e1,e2) - max(s1,s2) + 1)
                union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
                return inter / union if union>0 else 0

            def segment_f1_score(y_true, y_pred, iou_thr):
                gt_segs   = extract_segments(y_true)
                pred_segs = extract_segments(y_pred)

                # 按 label 分组
                gt_by_lab   = {}
                pred_by_lab = {}
                for lab,s,e in gt_segs:   gt_by_lab.setdefault(lab, []).append((s,e))
                for lab,s,e in pred_segs: pred_by_lab.setdefault(lab, []).append((s,e))

                TP = FP = FN = 0
                for lab, p_segs in pred_by_lab.items():
                    gt_list = gt_by_lab.get(lab, [])
                    for ps in p_segs:
                        # 找到同 label 下最大的 IoU
                        best_iou = max((segment_iou(ps, gs) for gs in gt_list), default=0)
                        if best_iou >= iou_thr:
                            TP += 1
                        else:
                            FP += 1
                for lab, g_segs in gt_by_lab.items():
                    p_list = pred_by_lab.get(lab, [])
                    for gs in g_segs:
                        best_iou = max((segment_iou(gs, ps) for ps in p_list), default=0)
                        if best_iou < iou_thr:
                            FN += 1

                prec = TP / (TP+FP) if TP+FP>0 else 0
                rec  = TP / (TP+FN) if TP+FN>0 else 0
                f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
                return f1

            iou_thresholds = [0.10, 0.25, 0.50]
            f1_seg = {}
            for thr in iou_thresholds:
                f1_seg[thr] = segment_f1_score(all_gts, all_preds, thr)

            # ===== 新增：Edit Distance Metric =====
            def levenshtein_norm(p, t):
                # 删除连续重复项
                def dedup(seq):
                    new = [seq[0]]
                    for s in seq[1:]:
                        if s != new[-1]:
                            new.append(s)
                    return new
                p, t = dedup(p), dedup(t)
                n, m = len(p), len(t)
                if n == 0: return 0.0
                D = np.zeros((n+1, m+1), dtype=np.uint16)
                for i in range(n+1): D[i][0] = i
                for j in range(m+1): D[0][j] = j
                for i in range(1, n+1):
                    for j in range(1, m+1):
                        cost = 0 if p[i-1] == t[j-1] else 1
                        D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+cost)
                norm_dist = 1 - D[n][m] / max(n, m)
                return norm_dist

            edit_score = levenshtein_norm(all_preds, all_gts)

            # ===== 打印所有指标 =====
            t1 = time.time()
            print(f"✔️ Prediction Finished in {t1-t0:.2f}s")
            print(f"📊 Frame-level Accuracy:          {acc:.4f}")
            print(f"📈 Frame-level F1 (macro):        {f1:.4f}")
            print(f"🔄 Frame-level Balanced-Acc:      {bal:.4f}")
            print(f"⭐ Frame-level mAP:               {mAP:.4f}")
            for thr, val in f1_seg.items():
                pct = int(thr*100)
                print(f"🎯 Segment-level F1 @ IoU {pct}%:   {val:.4f}")
            print(f"✏️ Segment-level Edit Score:      {edit_score:.4f}")

  
            # ===== Return metrics as dict =====
            metrics = {
                "frame_acc": acc,
                "frame_f1": f1,
                "frame_balanced_acc": bal,
                "frame_mAP": mAP,
                "edit_score": edit_score,
            }
            for thr, val in f1_seg.items():
                metrics[f"segment_f1_iou_{int(thr*100)}"] = val
        
            return metrics
     