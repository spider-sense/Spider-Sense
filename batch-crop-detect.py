# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:44:26 2021

@author: derph
"""

import argparse
import time
from pathlib import Path
import common

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

import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

import math

from predict import Predictor
from helperFunctions import distGet, getKeyPoints, getCropBoxes, bbox_overlap, medianCropWidth

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
           wsl=False, # option if WSL is being used 
           handheld=False # option for only detecting handheld classes
           ):
    # generating COCO maps
    category_name = ['frisbee', 'sports ball', 'baseball glove', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant', 'mouse', 'remote', 'cell phone', 'book', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    category_ids = [29, 32, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 58, 64, 65, 67, 73, 76, 77, 78, 79]
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
            myImg = im0s[0].copy()
        else:
            myImg = im0s.copy()
        myImg = letterbox(myImg, imgsz, stride=32)[0]
        keypoints, humans = getKeyPoints(myImg, e)
        cropBoxes = [getCropBoxes(point[0], myImg, 2, device, point[1]) for point in keypoints]       
        cropBoxes = [box for box in cropBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
        checkBoxes = [getCropBoxes(point[0], myImg, 1, device, point[1]) for point in keypoints]
        checkBoxes = [box for box in checkBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
        
        # optimizing crop boxes
        i = 0
        while i < len(cropBoxes) - 1:
            box1 = cropBoxes[i]
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            for j in range(i + 1, len(cropBoxes)):             
                # getting necessary info to determine crop information
                box2 = cropBoxes[j]
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                rectBox = [min(box1[0], box2[0]), min(box1[1], box2[1]), \
                           max(box1[2], box2[2]), max(box1[3], box2[3]), 0, 0]
                areaLarge = (rectBox[2] - rectBox[0]) * (rectBox[3] - rectBox[1])
                
                # doing checks to determine if boxes should be combined
                if area1 + area2 > areaLarge and (bbox_overlap(box1, box2) >= 0.4 or \
                                                  bbox_overlap(box2, box1) >= 0.4):
                    cropBoxes.append(rectBox)
                    cropBoxes.pop(j)
                    cropBoxes.pop(i)
                    i -= 1
                    break
            i += 1 
        
        # removing extra crop boxes
        cropBoxes = [i for i in cropBoxes if type(i) != bool]
            
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
    parser.add_argument('--handheld', default=False, action='store_true', help='if wsl is used then image not shown')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('tensorboard', 'thop'))

    detect(**vars(opt))