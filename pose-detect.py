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

import common
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

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
                #print("arm width")
                pass
            else:
                headWidth = False
                
                # moving onto upper bounding box
                upperBody = human.get_upper_body_box(432, 368)
                if type(upperBody) == dict and upperBody["w"] <= w/4:
                    headWidth = upperBody["w"]/2
                    #print("upper body width")
                    pass
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
    box = torch.Tensor([lowX, lowY, upX, upY, 0, 0]).to(device)
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
    interArea = abs(xB - xA) * abs(yB - yA)
     
    # box1 area
    box1Area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    
    # returning percentage of overlap if intersection is not 0
    if box1Area != 0:
        overlap = interArea / box1Area
        return overlap
    else:
        return 0

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
    # generating COCO category map
    category_name = ['tie', 'frisbee', 'sports ball', 'baseball glove', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant', 'mouse', 'remote', 'cell phone', 'book', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    category_ids = [27, 29, 32, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 58, 64, 65, 67, 73, 76, 77, 78, 79]
    handheld_map = {}
    for i in range(0, len(category_ids)):
        handheld_map[category_ids[i]] = category_name[i]
    print("map", handheld_map)
    
    # creating AI model things
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
        #myImg = predictor(myImg, None)
        keypoints, humans = getKeyPoints(myImg, e)
        cropBoxes = [getCropBoxes(point[0], myImg, 3, device, point[1]) for point in keypoints]       
        cropBoxes = [box for box in cropBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
        checkBoxes = [getCropBoxes(point[0], myImg, 0.75, device, point[1]) for point in keypoints]
        checkBoxes = [box for box in checkBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
                
        # if no crops then early exit
        if len(cropBoxes) == 0:
            print("Done Early:", time_sync()-t1)
            continue
        
        # Deblurring image
        timeDeblurOne = time_sync()
        img = myImg
        #img = predictor(myImg, None)
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        timeDeblurTwo = time_sync()
    
        # Actually doing the torch things
        #high.flops()
        time_one = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        print("nap time:", img.shape)
        epochs = 10000
        time_two = time_sync()
        for i in range(0, epochs):
            pred = model(img, augment=augment)[0]
        time_twoHalf = time_sync()
        for i in range(0, epochs):
            bread = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)        
        pred = bread
        time_three = time_sync()
        
        #print("Dataset Preparation:", time_two - time_one)
        #print("Readying keypoints:", timeDeblurOne - t1)
        #print("Deblurring:", timeDeblurTwo - timeDeblurOne)
        print("Inference:", (time_twoHalf - time_two) / epochs)
        print("NMS:", (time_three - time_twoHalf) / epochs)
        print("Inference + NMS:", (time_three - time_two) / epochs)
        #FLOPs = high.stop_counters()
        
        t2 = time_sync()
        
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size and same thing done for crops and check boxes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for i in range(0, len(cropBoxes)):
                    cropBoxes[i][0] = (cropBoxes[i][0]/myImg.shape[1] * im0.shape[1]).round()
                    cropBoxes[i][2] = (cropBoxes[i][2]/myImg.shape[1] * im0.shape[1]).round()
                    cropBoxes[i][1] = (cropBoxes[i][1]/myImg.shape[0] * im0.shape[0]).round()
                    cropBoxes[i][3] = (cropBoxes[i][3]/myImg.shape[0] * im0.shape[0]).round()
                for i in range(0, len(checkBoxes)):
                    checkBoxes[i][0] = (checkBoxes[i][0]/myImg.shape[1] * im0.shape[1]).round()
                    checkBoxes[i][2] = (checkBoxes[i][2]/myImg.shape[1] * im0.shape[1]).round()
                    checkBoxes[i][1] = (checkBoxes[i][1]/myImg.shape[0] * im0.shape[0]).round()
                    checkBoxes[i][3] = (checkBoxes[i][3]/myImg.shape[0] * im0.shape[0]).round()
                
                # Check if any overlap between keypoint and checkBoxes (handheld weapon)
                newDet = []
                for detection in det:
                    for crop in cropBoxes:
                        # check if detection in outCrop
                        if bbox_iou(detection, crop) > 0 and bbox_overlap(detection, crop) >= 0.6:
                            # checking if overlap between keypoint and cropBoxes
                            for check in checkBoxes:
                                checkOverlap = bbox_iou(detection, check)
                                if handheld_map.get(int(detection[5])) and checkOverlap > 0:
                                    newDet.append(detection)
                                    cv2.putText(im0, "Spider-Sense Tingling!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)                
                                    break
                            break

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string                
                            
                # Write results
                for *xyxy, conf, cls in reversed(newDet):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=5)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            
                # write keypoint boxes
                for *xyxy, conf, cls in reversed(cropBoxes):
                    c = int(cls)
                    if xyxy[3] - xyxy[1] > 0 and xyxy[2] - xyxy[0] > 0:
                        #save_one_box(xyxy, imc, file=save_dir/ 'wrist_crops' / names[c] / f'{p.stem}.jpg', BGR=True, pad=0)
                        plot_one_box(xyxy, im0, label="out", color=colors(c, True), line_thickness=5)
                for *xyxy, conf, cls in reversed(checkBoxes):
                    c = int(cls)
                    if xyxy[3] - xyxy[1] > 0 and xyxy[2] - xyxy[0] > 0:
                        #save_one_box(xyxy, imc, file=save_dir/ 'wrist_crops' / names[c] / f'{p.stem}.jpg', BGR=True, pad=0)
                        plot_one_box(xyxy, im0, label="in", color=colors(c, True), line_thickness=5)
                        
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        im0 = TfPoseEstimator.draw_humans(im0, humans, imgcopy=False)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

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