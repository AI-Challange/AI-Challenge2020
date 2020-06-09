# %%
import glob
import json
import os
import shutil
import operator
import sys
import argparse
import time
import math
import xml.etree.ElementTree as elemTree
import numpy as np
sys.path.append('/home/centos/anaconda3/envs/sh/lib')
from imantics import Polygons, Mask
from shapely.geometry import Polygon


MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

# make sure that the cwd() is the location of the python script (so that every path makes sense)
#os.chdir(os.path.dirname(os.path.abspath(__file__)))


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
    # let's sort the classes alphabetically
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
            ground_truth_data = gt_file_boxes[str(file_id)]
            ovmax = -1
            gt_match = -1
            ov = -1
            # load detected object bounding-box
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    p1 = Polygon(obj["poly"])
                    p1 = p1.buffer(0)
                    try:
                        p2 = Polygon(detection["poly"][0])
                        p2 = p2.buffer(0)
                        ov = p1.intersection(p2).area / (p1.area + p2.area - p1.intersection(p2).area)
                    except:
                        ov = -1
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
    print('mAP : ', mAP)
    return mAP

def evaluate(GT_PATH, DR_PATH) : #GT_PATH : root test folder DR_PATH : predict xml file path
    
    gt_file_boxes, gt_counter_per_class = read_test_file(GT_PATH)

    class_boxes = read_prediction_file(DR_PATH, gt_counter_per_class)

    return cal_mAP(gt_file_boxes, gt_counter_per_class, class_boxes)




def read_test_file(root):

    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_file_boxes = {}
    folder_list = glob.glob(root+'/*')
    for fpath in folder_list:
        filename = glob.glob(fpath + '/*.xml')
        labels = elemTree.parse(filename[0])
        labels = labels.findall('./image')

        for label in labels :
            file_id = label.attrib['name'].split(".jpg", 1)[0]
            bounding_boxes = []
            is_difficult = False
            already_seen_classes = []
            pos=[]

            for polygon_info in label.findall('./polygon') :
                class_name,_,points,_ = polygon_info.attrib.values()
                points = points.split(';')
                poly = []
                for p in points:
                    x , y = p.split(',')
                    poly.append([int(float(x)), int(float(y))])
                poly = np.array(poly, dtype=np.int64)

                if class_name != 'bike_lane' :
                    if len(polygon_info.findall('attribute')) == 0:
                        continue#                

                    class_name += '_'+polygon_info.findall('attribute')[0].text
                    if class_name == 'alleyblocks':
                        print(filename, file_id)
                pos.append({"class_name": class_name, "poly" : poly, "used" : False})

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
            pos  = np.array(pos)
            gt_file_boxes[file_id] = pos
    return gt_file_boxes, gt_counter_per_class

def read_prediction_file(file_path, gt_counter_per_class):

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)

    """
    detection-results
        Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    xml_path = glob.glob(file_path)
    labels = elemTree.parse(xml_path[0])
    labels = labels.findall('./image')
    class_boxes = {}
    for class_index, class_name in enumerate(gt_classes):
        polygons = []
        for label in labels :
            file_id = label.attrib['name']
            for pred in label.findall('./predict'):
                try:
                    temp_class_name, confidence, poly = pred.attrib['class_name'], pred.attrib['score'],pred.attrib['polygon']
                except ValueError:
                    error_msg = 'check data' + label.attrib['name']
                if temp_class_name == class_name:
                    pos = []
                    poly = poly.split(';')[:-1]
                    for p in poly:
                        x, y = p.split(',')
                        pos.append([int(x), int(y)])
                    pos = np.array([pos])
                    polygons.append({'confidence': confidence, 'file_id':file_id, 'poly': pos})
        polygons.sort(key=lambda x:float(x['confidence']), reverse=True)
        class_boxes[class_name] = polygons
    return class_boxes

def evaluation_metrics(GT_PATH, DR_PATH):
    return evaluate(GT_PATH, DR_PATH)


def main() :
    args = argparse.ArgumentParser()
    args.add_argument("--prediction_file", type=str, default= "predictions/prediction2.xml")
    config = args.parse_args()


    GT_PATH = os.path.join(os.getcwd(), 'Surface_1')
    DR_PATH = config.prediction_file
    print(evaluation_metrics(GT_PATH, DR_PATH))

if __name__ == '__main__' :
    start = time.time()
    main()
    print('time: ', time.time()-start)



