"""
Calculate Mean Average Precision (mAP) and F1 score for a set of bounding boxes corresponding.

Usage:
     pythons calculate_map_score_per_class.py --detections_record=detections_all_xviewz151617_dota.record --label_map=1class.pbtxt

"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
import matplotlib.pyplot as plt

from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

sns.set_style('white')
sns.set_context('poster')

flags = tf.app.flags

flags.DEFINE_string('label_map', None, 'Path to the label map')
flags.DEFINE_string('detections_record', None, 'Path to the detections record file')

FLAGS = flags.FLAGS

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

groundtruth_boxes_l = []
detection_boxes_l = []

TPS = []
FPS = []
TNS = []
FNS = []

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    TPS.append(tp)
    FPS.append(fp)
    FNS.append(fn)
    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def average_percision(precisions, recalls):
    """Calculate mean percision score from 11points interpretation

    Args:
        precisions (numpy array): Precision scores from the fixed prediction score and IOU threshods;
        recalls (numpy array): Recall scores from the fixed prediction score and IOU threshods.

    Returns:
        avg_prec (float): Average precision score at the fixed prediction score and IOU threshods.
    """
    prec_at_rec = []

    for recall_level in np.linspace(0.0, 1.0, 11):
        args = np.argwhere(recalls >= recall_level).flatten()
        for row in args:
            prec = precisions[row]
            prec_at_rec.append(prec)
    prec_at_rec = np.array(prec_at_rec)
    avg_prec = np.mean(prec_at_rec)
    return avg_prec

def get_avg_precision_at_iou(gtb, prdb, score, cls, score_thr, iou_thr=0.5): #calssjson, score_thr, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.
    Args:
        classjson(diction): diction that save class map
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: avg precision as well as summary info about the PR curve
        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    precisions_ = []
    recalls_ = []

    gt_boxes = gtb
    pred_boxes = prdb
    pred_boxes_thrsh = []
    for i in range(len(pred_boxes)):
        print(i)
        print(pred_boxes[i])
        print(score[i])
        if score[i] >= score_thr:
            pred_boxes_thrsh.append(pred_boxes[i])
    result = get_single_image_results(gt_boxes, pred_boxes_thrsh, iou_thr)
    img_result = dict(img_id = result)
    precision, recall = calc_precision_recall(img_result)
    precisions_.append(precision)
    recalls_.append(recall)

    precisions = np.array(precisions_)
    recalls = np.array(recalls_)
    avg_prec = average_percision(precisions, recalls)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': score_thr}

def get_score_high_avg_perc(score_threds, avg_percs):
    """Getting the prediction score threds that output the highest average percision score for the class
    Args:
        score_threds (numpy array): Numpy array at 10 steps range from 0.3 to 0.95;
        avg_percs (numpy array): the average persicion from the given pred score and iou = 0.5

    Returns:
        score: the pred score that get the highest average precision score.

    """
    ind = np.argmax(avg_percs)
    score = score_threds[ind]

    return '{:0.2f}'.format(score)

def plot_pr_curve(avg_precs, score_thrs, category, label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(score_thrs, avg_precs,label=label, s=20, color=color)
    ax.set_xlabel('pred score threshod')
    ax.set_ylabel('average precision score')
    ax.set_title('average precision score for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax


if __name__ == "__main__":
    required_flags = ['detections_record', 'label_map', 'output_path']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    label_map = label_map_util.load_labelmap(FLAGS.label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)

    label_map_dict = {'airplane':1}

    label_map_reves = {}
    for key, value in label_map_dict.items():
        label_map_reves[value] =key

    iou_thr = 0.5

    clses = [i for i in range(1, 1)]
    for i in range(1, 2):
        clt_json = f"out_cls{i}.json"
        category = label_map_reves[i]
        record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.detections_record)
        data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

        #confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))
        precisions_ = []
        recalls_ = []
        f1s = []
        start_time = time.time()
        ax = None
        avg_precs = []
        score_thrs = []
        category
        image_index = 0
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            decoded_dict = data_parser.parse(example)

            image_index += 1

            if decoded_dict:
                groundtruth_boxes = decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
                groundtruth_classes = decoded_dict[standard_fields.InputDataFields.groundtruth_classes]

                detection_scores = decoded_dict[standard_fields.DetectionResultFields.detection_scores]
                detection_classes = decoded_dict[standard_fields.DetectionResultFields.detection_classes][detection_scores >= CONFIDENCE_THRESHOLD]
                detection_boxes = decoded_dict[standard_fields.DetectionResultFields.detection_boxes][detection_scores >= CONFIDENCE_THRESHOLD]

                groundtruth_boxes_l.append(groundtruth_boxes)
                detection_boxes_l.append(detection_boxes)

                for idx, score_thr in enumerate(np.linspace(0.3, 0.95, 10)):
                    data = get_avg_precision_at_iou(groundtruth_boxes, detection_boxes, detection_scores, detection_classes, score_thr, iou_thr)
                    avg_precs.append(data['avg_prec'])
                    score_thrs.append(score_thr)
                    precisions = data['precisions']
                    recalls = data['recalls']
                    f1 = 2 * precisions[0] * recalls[0] / (precisions[0] + recalls[0])
                    f1s.append(f1)

        avg_precs = np.array(avg_precs)
        score_threds = np.array(score_thrs)
        f1s = [f1 for f1 in f1s if str(f1) != 'nan']
        f1s_arr = np.array(f1s)
        score= get_score_high_avg_perc(score_threds, avg_precs)
        ax = plot_pr_curve(
        avg_precs, score_threds, category, label='{:.2f}'.format(score_thr), color=COLORS[idx*2], ax=ax)

    avg_precs = [ap for ap in avg_precs]
    score_thrs = [thr for thr in score_thrs]
    f1s_l = [f for f in f1s]
    print('f1: {:.2f}'.format(np.mean(f1s)))
    print('map: {:.2f}'.format(np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('optimal prediction score threshod:', score)
    # print('pred_score_thrs:  ', score_thr)
    plt.legend(loc='upper right', title='Pred Score Thr', frameon=True)
    for xval in np.linspace(0.3, 0.95, 10):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    print("TPS: ", sum(TPS))
    print("FPS: ", sum(FPS))
    print("FNS: ", sum(FNS))
    plt.show()
