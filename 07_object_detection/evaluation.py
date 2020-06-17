import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import xml.etree.ElementTree as elemTree
import numpy as np


os.chdir(os.path.dirname(os.path.abspath(__file__)))


# throw error and exit
def error(msg):
    print(msg)
    sys.exit(0)


# check if the number is a float between 0.0 and 1.0

def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def voc_ap(rec, prec):

    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
 
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
 
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def cal_mAP(gt_file_boxes, gt_counter_per_class, class_boxes) :
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    
    sum_AP = 0.0

    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        
        dr_data = class_boxes[class_name]

        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            ground_truth_data = gt_file_boxes[file_id]
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [ float(x) for x in detection["bbox"].split() ]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [ float(x) for x in obj["bbox"].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # set minimum overlap
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
    mAP = sum_AP / n_classes
    return mAP

def evaluate(GT_PATH, DR_PATH) :
    
    gt_file_name = glob.glob(GT_PATH + '/*.xml')[0]
    gt_file_boxes, gt_counter_per_class = read_test_file(gt_file_name)

    class_boxes = read_prediction_file(DR_PATH, gt_counter_per_class)

    return cal_mAP(gt_file_boxes, gt_counter_per_class, class_boxes)

def read_prediction_file(xml_name, gt_counter_per_class):

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)

    # get a list with the detection-results files
    pred_file = elemTree.parse(xml_name)

    class_boxes = {}

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for image_info in pred_file.findall('image'):
            file_id = image_info.attrib['name']
            for pred_info in image_info.findall('predict'):
                try:
                    tmp_class_name, confidence, left, right, top, bottom = pred_info.attrib.values()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    if float(confidence) >= MINSCORE :
                        bbox = left + " " + top + " " + right + " " +bottom
                        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        class_boxes[class_name] = bounding_boxes
    return class_boxes

def read_test_file(file_name):

    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_file_boxes = {}

    labels = elemTree.parse(file_name)
    labels = labels.findall('./image')

    for label in labels :
        file_id = label.attrib['name'].split(".jpg", 1)[0]
        file_id = file_id.split(".png", 1)[0]
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []

        for box_info in label.findall('./box') :
            class_name,x1,y1,x2,y2 = box_info.attrib['label'],box_info.attrib['xtl'],box_info.attrib['ytl'],box_info.attrib['xbr'],box_info.attrib['ybr'] 
            x1,y1,x2,y2 = x1.split('.')[0],y1.split('.')[0],x2.split('.')[0],y2.split('.')[0]
        
            bbox = x1 + " " + y1 + " " + x2 + " " +y2
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
            
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)

        gt_file_boxes[file_id] = bounding_boxes
    return gt_file_boxes, gt_counter_per_class

def evaluation_metrics(GT_PATH, DR_PATH):
    return evaluate(GT_PATH, DR_PATH)

MINOVERLAP = 0.75 # default value (defined in the PASCAL VOC2012 challenge)
MINSCORE = 0.0

def main() :
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_path', type=str, default='./prediction/predictions.xml')
    args.add_argument('--test_path', type=str, default='data/test_admin')
    
    config = args.parse_args()

    print(evaluation_metrics(config.test_path, config.prediction_path))

if __name__ == '__main__' :
    main()

