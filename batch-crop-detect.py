# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:44:26 2021

@author: derph
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.metrics import bbox_iou
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.augmentations import letterbox

import argparse
import logging
import time
import ast

import tf_pose.common
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

import math
from pypapi import events, papi_high as high

from predict import Predictor

"""
basic function to get distance between two points
"""
def distGet(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

"""
This is the function I made to get the keypoints.  I'll be taking a crop at
each of these keypoints and feeding it to the actual object detector.  The 
keypoints will preferably be the wrists but if no wrists are detected it will
settle for the shoulders
"""
def getKeyPoints(img, e):    
    # Running inference
    print(img.shape)
    w = img.shape[1]
    h = img.shape[0]
    humans = e.inference(img, scales=[None])
    #print("humans", humans)
    
    # Getting keypoints
    KP = []
    for human in humans:
        parts = human.body_parts
        head = human.get_face_box(432, 368)
        if type(head) == dict and head["w"] <= w/8:
            headWidth = head["w"]
        else:
            # change to forearm length
            leftForearm = -1
            if 7 in parts and 6 in parts:
                leftForearm = distGet((parts[6].x * w, parts[6].y * h), (parts[7].x * w, parts[7].y * h))
            rightForearm = -1
            if 3 in parts and 4 in parts:
                rightForearm = distGet((parts[3].x * w, parts[3].y * h), (parts[4].x * w, parts[4].y * h))
            
            # forearm scenarios
            headWidth = False
            if leftForearm != -1:
                headWidth = leftForearm
            elif rightForearm != -1:
                headWidth = rightForearm
            elif leftForearm != 1 and rightForearm != 1:
                headWidth = (leftForearm + rightForearm) / 2
            
            # checking if forearm width is valid
            if type(headWidth) != bool and headWidth <= w/8:
                pass
            else:
                headWidth = False
                
                # moving onto upper bounding box
                upperBody = human.get_upper_body_box(432, 368)
                if type(upperBody) == dict and upperBody["w"] <= w/4:
                    headWidth = upperBody["w"]/2
                else:
                    headWidth = w/8
                
        # Searching for wrists
        if 7 in parts:
            KP.append([parts[7], headWidth])
        if 4 in parts:
            KP.append([parts[4], headWidth])
                 
    return KP, humans

"""
Function for getting the crop of an image.  Will look at headWidth of img.
"""
def getCropBoxes(point, img, factor, device, cropWidth):
    cropWidth *= factor
    pointX = round(img.shape[1] * point.x)
    pointY = round(img.shape[0] * point.y)
    lowX = pointX - cropWidth
    upX = pointX + cropWidth
    lowY = pointY - cropWidth
    upY = pointY + cropWidth
    # maintaining aspect ratio if hits a border
    if lowX < 0:
        off = 0 - lowX
        lowX = 0
        upX + off
    if lowY < 0:
        off = 0 - lowY
        lowY = 0
        upY = lowY + off
    box = [lowX, lowY, upX, upY, 0, 0]
    return box

"""
gets overlap of box1 into box2 (not same as IOU) 
"""
def bbox_overlap(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    # intersection area
    interArea = abs(xB - xA + 1) * abs(yB - yA + 1)
     
    # box1 area
    box1Area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    
    # returning percentage of overlap if intersection is not 0
    if box1Area != 0:
        overlap = interArea / box1Area
        return overlap
    else:
        return 0

"""
gets median width in crop boxes
"""
def medianCropWidth(cropBoxes, img):
    # getting median width
    medianIndex = math.floor(len(cropBoxes)/2)
    middleWidth = max(cropBoxes[medianIndex][3]-cropBoxes[medianIndex][1], cropBoxes[medianIndex][2]-cropBoxes[medianIndex][0])
    middleWidth = middleWidth - (middleWidth % 32)
    
    # if middleWidth is 0 then just adding 32 and returning
    if middleWidth == 0:
        middleWidth += 32
        return middleWidth
    
    # otherwise comparing middle width with entire image shape to ensure pixel advantage
    totalPixels = img.shape[0] * img.shape[1]
    cropPixels = middleWidth * middleWidth * len(cropBoxes)
    
    # lowering crop if it's creating more pixels than total image
    while cropPixels >= totalPixels:
        middleWidth -= 32
        cropPixels = middleWidth * middleWidth * len(cropBoxes)
      
    # if middleWidth is 0 then just adding 32
    if middleWidth == 0:
        middleWidth += 32    
    
    return middleWidth

@torch.no_grad()
def detect(model="mobilenet_thin", # A model option for being cool
           weights='yolov5s.pt',  # model.pt path(s)
           weights_path='fpn_mobilenet.h5', # deblurrer path
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           wsl=False # option if WSL is being used 
           ):
    # generating COCO maps
    category_name = ['tie', 'frisbee', 'sports ball', 'baseball glove', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant', 'mouse', 'remote', 'cell phone', 'book', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    category_ids = [27, 29, 32, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 58, 64, 65, 67, 73, 76, 77, 78, 79]
    handheld_map = {}
    for i in range(0, len(category_ids)):
        handheld_map[category_ids[i]] = category_name[i]
    
    # generating AI models
    w, h = 432, 368
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    predictor = Predictor(weights_path=weights_path)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(project)
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        if not wsl:
            view_img = check_imshow()
        else:
            view_img = False
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # starting time sync early due to openpose
        t1 = time_sync()
        
        # Openpose getting keypoints and individual crops
        if type(im0s) == list:
            myImg = im0s[0]
        else:
            myImg = im0s
        myImg = letterbox(myImg, imgsz, stride=32)[0]
        keypoints, humans = getKeyPoints(myImg, e)
        cropBoxes = [getCropBoxes(point[0], myImg, 2.5, device, point[1]) for point in keypoints]       
        cropBoxes = [box for box in cropBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
        checkBoxes = [getCropBoxes(point[0], myImg, 0.75, device, point[1]) for point in keypoints]
        checkBoxes = [box for box in checkBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
        """
        # optimizing crop boxes
        i = 0
        while i < len(cropBoxes) - 1:
            # False check
            if type(cropBoxes[i]) == bool:
                i += 1
                continue
            for j in range(i + 1, len(cropBoxes)):
                # False check for other box
                if type(cropBoxes[j]) == bool:
                    continue
                
                # getting necessary info to determine crop information
                box1, box2 = cropBoxes[i], cropBoxes[j]
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                rectBox = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3]), 0, 0]
                areaLarge = (rectBox[2] - rectBox[0]) * (rectBox[3] - rectBox[1])
                
                # doing checks specified in paper to determine if rectBox should be pursued
                if area1 + area2 > areaLarge and (bbox_overlap(box1, box2) >= 0.4 or bbox_overlap(box2, box1) >= 0.4):
                    cropBoxes[i], cropBoxes[j] = False, False
                    cropBoxes.append(rectBox)
                    break
            i += 1
        """
        # removing extra crop boxes
        for i in range(0, cropBoxes.count(False)):
            cropBoxes.remove(False)
            
        # if no crops then early exit otherwise getting crop images
        if len(cropBoxes) == 0:
            print("Done Early:", time_sync()-t1)
            continue

        # sorting crop boxes to determine collage size
        cropBoxes.sort(key = lambda x: max(x[3]-x[1], x[2]-x[0]))
        medianWidth = medianCropWidth(cropBoxes, myImg)
        cropBoxes = [torch.Tensor(i).to(device) for i in cropBoxes]
        checkBoxes = [torch.Tensor(i).to(device) for i in checkBoxes]
        
        cropImages = [save_one_box(box[:4], myImg, BGR=True, save=False) for box in cropBoxes]         
        medianWidth = round(medianWidth)
        
        # actual inference + NMS       
        crops = []
        for i in range(0, len(cropImages)):
            # doing dataseet changes
            crop = letterbox(cropImages[i], medianWidth, stride=32)[0]
            crop = crop.transpose((2, 0, 1))[::-1]
            crop = np.ascontiguousarray(crop)
            crop = crop.astype(float)
            crops.append(crop)
        crops = np.array(crops)
        crops /= 255.0
        crops = torch.from_numpy(crops).to(device)
        crops = crops.half() if half else crops.float()
        
        print("nap time:", crops.shape)
        epochs = 1
        t_two = time_sync()
        for i in range(0, epochs):
            preds = model(crops, augment=augment)[0]
        t_twoHalf = time_sync()
        for i in range(0, epochs):
            bread = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred = bread
        t2 = time_sync()
        print("Inference:", (t_twoHalf - t_two) / epochs)
        print("NMS:", (t2 - t_twoHalf) / epochs)
        print("Inference + NMS:", (t2 - t_two) / epochs)
        #FLOPs = high.stop_counters()
        FLOPs = 69
        
        # Print time (inference + NMS)
        #print(f'Done. ({t2 - t_one:.3f}s) ({t_two - t_one:.3f}s) ({t2 - t1:.3f}s) {FLOPs} FLOPs')
        
        # Saving bounding boxes
        for im, pred in enumerate(preds):
            break
            cropBox = cropBoxes[im]
            for i, det in enumerate(pred):
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                    
                for i in range(0, len(det)):
                    pass
                
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                #s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop       

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--weights_path', default=str, help='weight path for DeblurGANv2')    
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--wsl', default=False, action='store_true', help='if wsl is used then image not shown')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('tensorboard', 'thop'))

    detect(**vars(opt))