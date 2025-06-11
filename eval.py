import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os
from sklearn.metrics import average_precision_score

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content
 
 
def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends
 
 
def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], dtype=float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
     
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
 
    return score
 
 
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)
 
 
def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)
 
def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.get_cmap('seismic')
    # color_map =
    fig = plt.figure(figsize=(15, num_pics * 1.5))
 
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=20)
 
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
 
    plt.close()
 
 
def segment_bars_with_confidence(save_path, confidence, *labels):
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
 
 
def func_eval(dataset, recog_path, file_list):
    ground_truth_path = "./data/" + dataset + "/groundtruth/"
    mapping_file = "./data/" + dataset + "/mapping.txt"
    list_of_videos = read_file(file_list).split('\n')[:-1]
 
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
 
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0
    frame_map_all = 0.
 
    for vid in list_of_videos:
 
         
        gt_file = os.path.join(ground_truth_path, vid)
        if not os.path.exists(gt_file):
            if os.path.exists(gt_file + ".txt"):
                gt_file += ".txt"
            else:
                raise FileNotFoundError(f"Ground truth file not found: {gt_file} or {gt_file}.txt")
        gt_content = read_file(gt_file).split('\n')[0:-1]
 
        # recognition file could be directly under recog_path or in split_x subfolder
        vid_name = vid.split('.')[0]
        # try direct path
        recog_file_base = os.path.join(recog_path, vid_name)
        if os.path.exists(recog_file_base) or os.path.exists(recog_file_base + '.txt'):
            recog_file = recog_file_base
        else:
            # search in split_* subdirectories
            recog_file = None
            for sub in os.listdir(recog_path):
                subdir = os.path.join(recog_path, sub)
                if os.path.isdir(subdir) and sub.startswith('split_'):
                    candidate = os.path.join(subdir, vid_name)
                    if os.path.exists(candidate) or os.path.exists(candidate + '.txt'):
                        recog_file = candidate
                        break
            if recog_file is None:
                raise FileNotFoundError(f"Recognition file not found for video {vid_name} in {recog_path} or its split_ subfolders")
        # ensure .txt extension
        if not os.path.exists(recog_file):
            recog_file += '.txt'

        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    # --- Frame-level mAP 计算 ---
    all_labels = []
    all_preds = []

    for vid in list_of_videos:
        gt_file = os.path.join(ground_truth_path, vid)
        if not os.path.exists(gt_file):
            if os.path.exists(gt_file + ".txt"):
                gt_file += ".txt"
            else:
                raise FileNotFoundError(f"Ground truth file not found: {gt_file} or {gt_file}.txt")
        gt_content = read_file(gt_file).split('\n')[0:-1]
        # recognition file could be directly under recog_path or in split_x subfolder
        vid_name = vid.split('.')[0]
        # try direct path
        recog_file_base = os.path.join(recog_path, vid_name)
        if os.path.exists(recog_file_base) or os.path.exists(recog_file_base + '.txt'):
            recog_file = recog_file_base
        else:
            # search in split_* subdirectories
            recog_file = None
            for sub in os.listdir(recog_path):
                subdir = os.path.join(recog_path, sub)
                if os.path.isdir(subdir) and sub.startswith('split_'):
                    candidate = os.path.join(subdir, vid_name)
                    if os.path.exists(candidate) or os.path.exists(candidate + '.txt'):
                        recog_file = candidate
                        break
            if recog_file is None:
                raise FileNotFoundError(f"Recognition file not found for video {vid_name} in {recog_path} or its split_ subfolders")
        # ensure .txt extension
        if not os.path.exists(recog_file):
            recog_file += '.txt'

        recog_content = read_file(recog_file).split('\n')[1].split()

        all_labels.extend(gt_content)
        all_preds.extend(recog_content)

    # ensure all_labels and all_preds have same length for mAP
    if len(all_labels) != len(all_preds):
        print(f"[WARNING] Misaligned lengths: all_labels={len(all_labels)}, all_preds={len(all_preds)}, truncating to minimum")
        min_len = min(len(all_labels), len(all_preds))
        all_labels = all_labels[:min_len]
        all_preds = all_preds[:min_len]

    label_set = sorted(list(actions_dict.keys()))
    y_true = np.array([[1 if l == label else 0 for label in label_set] for l in all_labels])
    y_pred = np.array([[1 if p == label else 0 for label in label_set] for p in all_preds])

    frame_map = average_precision_score(y_true, y_pred, average='macro') * 100
    print("Frame mAP: %.4f" % frame_map)

    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(list_of_videos)
    print("Acc: %.4f" % (acc))
    print('Edit: %.4f' % (edit))
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
#         print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1

    return acc, edit, f1s, frame_map

def main():
    cnt_split_dict = {
        '50salads': 5,
        'gtea': 5,
        'breakfast': 4,
        'surgery_I3D': 5,
        'surgery_bridge':5  
    }
    
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--result_dir', default='results')


    args = parser.parse_args()

    acc_all = 0.
    edit_all = 0.
    f1s_all = [0.,0.,0.]
    frame_map_all = 0.
    
    # Always evaluate using the single test.bundle file
    recog_path = os.path.join(args.result_dir, args.dataset)
    file_list = f"./data/{args.dataset}/splits/test.bundle"
    acc_all, edit_all, f1s_all, frame_map_all = func_eval(
        args.dataset, recog_path, file_list)
    print("Acc: %.4f  Edit: %.4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
    print("Frame mAP: %.4f" % frame_map_all)


if __name__ == '__main__':
    main()
# #!/usr/bin/python2.7
# # adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

# import numpy as np
# import argparse
# import os


# def read_file(path):
#     with open(path, 'r') as f:
#         content = f.read()
#         f.close()
#     return content


# def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
#     labels = []
#     starts = []
#     ends = []
#     last_label = frame_wise_labels[0]
#     if frame_wise_labels[0] not in bg_class:
#         labels.append(frame_wise_labels[0])
#         starts.append(0)
#     for i in range(len(frame_wise_labels)):
#         if frame_wise_labels[i] != last_label:
#             if frame_wise_labels[i] not in bg_class:
#                 labels.append(frame_wise_labels[i])
#                 starts.append(i)
#             if last_label not in bg_class:
#                 ends.append(i)
#             last_label = frame_wise_labels[i]
#     if last_label not in bg_class:
#         ends.append(i)
#     return labels, starts, ends


# def levenstein(p, y, norm=False):
#     m_row = len(p)
#     n_col = len(y)
#     D = np.zeros([m_row+1, n_col+1], np.float)
#     for i in range(m_row+1):
#         D[i, 0] = i
#     for i in range(n_col+1):
#         D[0, i] = i

#     for j in range(1, n_col+1):
#         for i in range(1, m_row+1):
#             if y[j-1] == p[i-1]:
#                 D[i, j] = D[i-1, j-1]
#             else:
#                 D[i, j] = min(D[i-1, j] + 1,
#                               D[i, j-1] + 1,
#                               D[i-1, j-1] + 1)

#     if norm:
#         score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
#     else:
#         score = D[-1, -1]

#     return score


# def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
#     P, _, _ = get_labels_start_end_time(recognized, bg_class)
#     Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
#     return levenstein(P, Y, norm)


# def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
#     p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
#     y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

#     tp = 0
#     fp = 0

#     hits = np.zeros(len(y_label))

#     for j in range(len(p_label)):
#         intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
#         union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
#         IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
#         # Get the best scoring segment
#         idx = np.array(IoU).argmax()

#         if IoU[idx] >= overlap and not hits[idx]:
#             tp += 1
#             hits[idx] = 1
#         else:
#             fp += 1
#     fn = len(y_label) - sum(hits)
#     return float(tp), float(fp), float(fn)


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--dataset', default="gtea")
#     parser.add_argument('--split', default='1')

#     args = parser.parse_args()

#     ground_truth_path = "./data/"+args.dataset+"/groundTruth/"
#     recog_path = "./results/"+args.dataset+"/split_"+args.split+"/"
#     file_list = "./data/"+args.dataset+"/splits/teste.split"+args.split+".bundle"

#     list_of_videos = read_file(file_list).split('\n')[:-1]

#     overlap = [.1, .25, .5]
#     tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

#     correct = 0
#     total = 0
#     edit = 0

#     for vid in list_of_videos:
#         # Always append .txt for ground truth files
#         gt_file = os.path.join(ground_truth_path, vid + ".txt")
#         gt_content = read_file(gt_file).split('\n')[0:-1]

   
     
#         recog_file = os.path.join(recog_path, vid.split('.')[0] + ".txt")
#         recog_content = read_file(recog_file).split('\n')[1].split()

#         for i in range(len(gt_content)):
#             total += 1
#             if gt_content[i] == recog_content[i]:
#                 correct += 1

#         edit += edit_score(recog_content, gt_content)

#         for s in range(len(overlap)):
#             tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
#             tp[s] += tp1
#             fp[s] += fp1
#             fn[s] += fn1

#     print("Acc: %.4f" % (100*float(correct)/total))
#     print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
#     acc = (100*float(correct)/total)
#     edit = ((1.0*edit)/len(list_of_videos))
#     for s in range(len(overlap)):
#         precision = tp[s] / float(tp[s]+fp[s])
#         recall = tp[s] / float(tp[s]+fn[s])

#         f1 = 2.0 * (precision*recall) / (precision+recall)

#         f1 = np.nan_to_num(f1)*100
#         print('F1@%0.2f: %.4f' % (overlap[s], f1))

# if __name__ == '__main__':
#     main()
