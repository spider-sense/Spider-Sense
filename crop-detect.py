# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:44:26 2021

@author: derph
"""

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.metrics import bbox_iou
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.augmentations import letterbox

import argparse
import logging
import time
import ast

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

import math

from predict import Predictor
from helperFunctions import distGet, getKeyPoints, getCropBoxes, bbox_overlap, medianCropWidth

@torch.no_grad()
def detect(model="mobilenet_thin", # A model option for being cool
           weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
           weights_path='fpn_mobilenet.h5', # deblurrer path
           source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
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
           project=ROOT / 'runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           wsl=False, # option if WSL is being used 
           handheld=False, # option for only detecting handheld classes
           noDeblur=False, # option for whether or not to deblur
           noElbow=False, # option for whether or not to detect elbows
           innerRatio=1, # inner crop ratio
           outerRatio=2 # outer crop ratio 
           ):
    # generating COCO maps
    category_name = ['frisbee', 'sports ball', 'baseball glove', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant', 'mouse', 'remote', 'cell phone', 'book', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    category_ids = [29, 32, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 58, 64, 65, 67, 73, 76, 77, 78, 79]
    handheld_map = {}
    for i in range(0, len(category_ids)):
        handheld_map[category_ids[i]] = category_name[i]
    
    # generating AI models
    dim = [int(i) for i in model.split("_")[-1].split("x")]
    w, h = dim[0], dim[1]
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
        keypoints, humans = getKeyPoints(myImg, e, noElbow)
        outerBoxes = [getCropBoxes(point[0], myImg, outerRatio, device, point[1]) for point in keypoints]       
        outerBoxes = [box for box in outerBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
        cropBoxes = [getCropBoxes(point[0], myImg, outerRatio, device, point[1]) for point in keypoints]       
        cropBoxes = [box for box in cropBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
        checkBoxes = [getCropBoxes(point[0], myImg, innerRatio, device, point[1]) for point in keypoints]
        checkBoxes = [box for box in checkBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
        
        # optimizing crop boxes
        i = 0
        while i < len(outerBoxes) - 1:
            box1 = outerBoxes[i]
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            for j in range(i + 1, len(outerBoxes)):             
                # getting necessary info to determine crop information
                box2 = outerBoxes[j]
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                rectBox = [min(box1[0], box2[0]), min(box1[1], box2[1]), \
                           max(box1[2], box2[2]), max(box1[3], box2[3]), 0, 0]
                areaLarge = (rectBox[2] - rectBox[0]) * (rectBox[3] - rectBox[1])
                
                # doing checks to determine if boxes should be combined
                if area1 + area2 > areaLarge and (bbox_overlap(box1, box2) >= 0.4 or \
                                                  bbox_overlap(box2, box1) >= 0.4):
                    outerBoxes.append(rectBox)
                    outerBoxes.pop(j)
                    outerBoxes.pop(i)
                    i -= 1
                    break
            i += 1      
            
        # if no crops then early exit otherwise getting crop images
        if len(outerBoxes) == 0:
            print("Done Early:", time_sync()-t1)
            continue

        # sorting crop boxes to determine collage size
        outerBoxes.sort(key = lambda x: max(x[3]-x[1], x[2]-x[0]))
        medianWidth = medianCropWidth(outerBoxes, myImg)
        outerBoxes = [torch.Tensor(i).to(device) for i in outerBoxes]
        cropBoxes = [torch.Tensor(i).to(device) for i in cropBoxes]
        checkBoxes = [torch.Tensor(i).to(device) for i in checkBoxes]
        
        cropImages = [save_one_box(box[:4], myImg, BGR=True, save=False) for box in outerBoxes]
        
        # generating one big image containing all of the crops plus cropPad tracker
        time_one = time_sync()
        crops = []
        batchCrops = []
        cropPads = []
        medianWidth = round(medianWidth)
        for i in range(0, len(cropImages)):
            """
            batchCrop = letterbox(cropImages[i], medianWidth, stride=32)[0]
            x, y = 0, 0
            if batchCrop.shape[0] < medianWidth:
                y = medianWidth - batchCrop.shape[0]
                batchCrop = np.vstack((batchCrop, np.zeros((y, medianWidth, 3))))
            elif batchCrop.shape[1] < medianWidth:
                x = medianWidth - batchCrop.shape[1]
                batchCrop = np.hstack((batchCrop, np.zeros((medianWidth, x, 3))))
            batchCrop = batchCrop.transpose((2, 0, 1))[::-1]
            batchCrop = np.ascontiguousarray(batchCrop)
            batchCrop = batchCrop.astype(float)
            batchCrops.append(batchCrop)            
            """           
            crop = letterbox(cropImages[i], medianWidth, stride=32)[0]
            x, y = 0, 0
            if crop.shape[0] < medianWidth:
                y = medianWidth - crop.shape[0]
                crop = np.vstack((crop, np.zeros((y, medianWidth, 3))))
            elif crop.shape[1] < medianWidth:
                x = medianWidth - crop.shape[1]
                crop = np.hstack((crop, np.zeros((medianWidth, x, 3))))
             
            cropPads.append((x, y))
            crops.append(crop)
        """
        # creating batch crop input
        batchCrops = predictor(batchCrops, None)
        batchCrops = torch.from_numpy(np.array(batchCrops)).to(device)
        batchCrops = batchCrops.half() if half else batchCrops.float()
        if not noDeblur:
            batchCrops = predictor(batchCrops, None)
        batchTest = model(batchCrops, augment=augment)[0]
        """
        # Actually doing the torch things for our image but not before running deblur
        bigIm = np.vstack(tuple(crops))
        
        timeDeblurOne = time_sync()
        if not noDeblur:
            bigIm = predictor(bigIm, None)
        timeDeblurTwo = time_sync()
        #cv2.imwrite("./runs/detect/wrist_crops.jpg", bigIm)
        bigIm = bigIm.transpose((2, 0, 1))[::-1]
        bigIm = np.ascontiguousarray(bigIm)
        bigIm = torch.from_numpy(bigIm).to(device)
        bigIm = bigIm.half() if half else bigIm.float()  # uint8 to fp16/32
        bigIm /= 255.0  # 0 - 255 to 0.0 - 1.0
        if bigIm.ndimension() == 3:
            bigIm = bigIm.unsqueeze(0)
          
        # running actual inference + NMS
        print("nap time:", bigIm.shape)
        epochs = 1
        time_two = time_sync()
        for i in range(0, epochs):
            pred = model(bigIm, augment=augment)[0]
        time_twoHalf = time_sync()
        for i in range(0, epochs):
            bread = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)        
        pred = bread
        time_three = time_sync()

        #print("getting keypoints", time_one - t1)     
        #print("Deblurring time:", timeDeblurTwo - timeDeblurOne)
        #print("Dataset Preparation:", time_two - time_one - timeDeblurTwo + timeDeblurOne)
        print("Inference:", (time_twoHalf - time_two) / epochs)
        print("NMS:", (time_three - time_twoHalf) / epochs)
        print("Inference + NMS:", (time_three - time_two) / epochs)
        t2 = time_sync()
        print(f'Done. ({t2 - t1:.3f}s)')

        # drawing detections
        for j, det in enumerate(pred):
            # filtering detections so that detections mixed between images get yote
            det = [i.cpu().numpy() for i in det if i[3]-i[1] <= medianWidth and i[2]-i[0] <=medianWidth and not (math.floor(i[3]/medianWidth)*medianWidth>i[1] and abs((i[1]+(i[3]-i[1])/2) - math.floor(i[3]/medianWidth)*medianWidth) <= 15)]
            
            # readying sum variables
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[j], f'{j}: ', im0s[j].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) 
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % bigIm.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            # resizing check, crop, and outer boxes
            for i in range(0, len(outerBoxes)):
                outerBoxes[i][0] = (outerBoxes[i][0]/myImg.shape[1] * im0.shape[1]).round()
                outerBoxes[i][2] = (outerBoxes[i][2]/myImg.shape[1] * im0.shape[1]).round()
                outerBoxes[i][1] = (outerBoxes[i][1]/myImg.shape[0] * im0.shape[0]).round()
                outerBoxes[i][3] = (outerBoxes[i][3]/myImg.shape[0] * im0.shape[0]).round()
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
            
            # if any boxes left then converting them to the actual image
            if len(det):
                for i in range(0, len(det)):
                    # finding respective crop box
                    centerDetY = det[i][1] + (det[i][3] - det[i][1]) / 2
                    boxIndex = math.floor(centerDetY / medianWidth)
                    cropBox = outerBoxes[boxIndex]
                    pad = cropPads[boxIndex]
                    
                    # reformatting detection to be relative medianWidth by medianWidth square
                    floor = centerDetY - (centerDetY % medianWidth)
                    if det[i][1] <= floor:
                        det[i][1] = 0
                    else:
                        det[i][1] -= floor
                    det[i][3] -= floor
                    
                    # adjusting detection in case it overlaps with pads
                    if det[i][3] >= medianWidth - pad[1]:
                        det[i][3] = medianWidth - pad[1]
                    if det[i][2] >= medianWidth - pad[0]:
                        det[i][2] = medianWidth - pad[0]
                    
                    # reformatting detection to be relative to wrist crop bounding box in small image
                    width = (cropBox[2] - cropBox[0]).cpu().numpy()
                    height = (cropBox[3] - cropBox[1]).cpu().numpy()
                    det[i] = torch.Tensor([cropBox[0]+round(width*(det[i][0]/(medianWidth-pad[0]))), cropBox[1]+round(height*(det[i][1]/(medianWidth-pad[1]))), cropBox[0]+round(width*(det[i][2]/(medianWidth-pad[0]))), cropBox[1]+round(height*(det[i][3]/(medianWidth-pad[1]))), det[i][4], det[i][5]]).to(device)
                
                # Check if any overlap between checkBoxes and det (handheld weapon)
                buttDet = []
                newDet = []
                for detection in det:
                    for crop in cropBoxes:
                        # check if detection in outCrop
                        if handheld_map.get(int(detection[5])):
                            buttDet.append(detection)
                        if bbox_iou(detection, crop) > 0 and bbox_overlap(detection, crop) >= 0.6:
                            # checking if overlap between keypoint and cropBoxes
                            for check in checkBoxes:
                                checkOverlap = bbox_iou(detection, check)
                                if (not handheld or handheld_map.get(int(detection[5]))) and checkOverlap > 0:
                                    newDet.append(detection)
                                    cv2.putText(im0, "Spider-Sense Tingling!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)                
                                    break
                            break
                        
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
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
            # write all keypoint boxes and draw humans
            for *xyxy, conf, cls in reversed(outerBoxes):
                c = int(cls)
                if xyxy[3] - xyxy[1] > 0 and xyxy[2] - xyxy[0] > 0:
                    #save_one_box(xyxy, imc, file=save_dir/ 'wrist_crops' / names[c] / f'{p.stem}.jpg', BGR=True, pad=0)
                        annotator.box_label(xyxy, color=colors(c, True))
            for *xyxy, conf, cls in reversed(cropBoxes):
                c = int(cls)
                if xyxy[3] - xyxy[1] > 0 and xyxy[2] - xyxy[0] > 0:
                    #save_one_box(xyxy, imc, file=save_dir/ 'wrist_crops' / names[c] / f'{p.stem}.jpg', BGR=True, pad=0)
                    annotator.box_label(xyxy, color=colors(c, True))
            for *xyxy, conf, cls in reversed(checkBoxes):
                c = int(cls)
                if xyxy[3] - xyxy[1] > 0 and xyxy[2] - xyxy[0] > 0:
                    #save_one_box(xyxy, imc, file=save_dir/ 'wrist_crops' / names[c] / f'{p.stem}.jpg', BGR=True, pad=0)
                    annotator.box_label(xyxy, color=colors(c, True))
            im0 = TfPoseEstimator.draw_humans(im0, humans, imgcopy=False)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
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
    parser.add_argument('--handheld', default=False, action='store_true', help='if wsl is used then image not shown')    
    parser.add_argument('--noDeblur', default=False, action='store_true', help='option for disabling deblur')
    parser.add_argument('--noElbow', default=False, action='store_true', help='option for disabling elbow check')
    parser.add_argument('--innerRatio', default=1, type=int, help='inner crop ratio for pipeline')
    parser.add_argument('--outerRatio', default=2, type=int, help='outer crop ratio for pipeline')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('tensorboard', 'thop'))

    detect(**vars(opt))