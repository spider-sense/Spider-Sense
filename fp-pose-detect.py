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
from poseEstimation.infer import Pose
from poseEstimation.pose.utils.utils import draw_keypoints


@torch.no_grad()
def detect(model="mobilenet_thin", # A model option for being cool
           weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
           det_model=ROOT/ 'crowdhuman_yolov5m.pt', # detection model for pose
           pose_model=ROOT/ 'poseEstimation/simdr_hrnet_w48_256x192.pth', # pose model
           weights_path='fpn_mobilenet.h5', # deblurrer path
           source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           pose_conf_thres=0.4, # pose confidence threshold
           conf_thres=0.25,  # confidence threshold
           upper_conf_thres=1.1, # confidence threshold for ignoring pipeline
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
           visualize=False,  # visualize features
           update=False,  # update all models
           project=ROOT / 'runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           dnn=False,  # use OpenCV DNN for ONNX inference
           wsl=False, # option if WSL is being used 
           handheld=False, # option for only detecting handheld classes
           noDeblur=False, # option for whether or not to deblur
           noElbow=False, # option for whether or not to detect elbows
           noPose=False, # option for not showing the pose
           allDet=False, # option for showing all detections and not just 
           outerDet=True, # option for using outer det vs outer crop width
           innerRatio=1, # inner crop ratio
           outerRatio=2, # outer crop ratio
           poseNum=3 # number of detections to switch pose estimator
           ):
    
    # Getting the False Positives
    import json
    with open ("falsePositives.json", 'r') as file:
        falsePositives = json.loads(file.read())
    #############################
    
    # Getting respective false positive directory
    modelName = "det"
    falsePositives = falsePositives[modelName]
    #############################################
    
    # generating COCO category map
    handheld_map = {29: 'frisbee', 32: 'sports ball', 35: 'baseball glove', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 58: 'potted plant', 64: 'mouse', 65: 'remote', 67: 'cell phone', 73: 'book', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}   
    print("map", handheld_map)
    print("SAVED", project)
    
    # creating AI model things
    print(model)
    dim = [int(i) for i in model.split("_")[-1].split("x")]
    w, h = dim[0], dim[1]
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    pose = Pose(
        det_model,
        pose_model[0],
        imgsz,
        pose_conf_thres,
        iou_thres
    )
    predictor = Predictor(weights_path=weights_path)
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            # check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *[imgsz, imgsz]).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    t0 = time_sync()
    for path, img, im0s, vid_cap in dataset:
        # False positive key path make
        newPath = path.split("/")
        newPath = newPath[-1].split(".")[0]
        fpLabels = falsePositives.get(newPath)
        if fpLabels == None:
            fpLabels = []
        ##############################
        
        # starting time sync early due to openpose
        t1 = time_sync()
        
        # Actually doing the torch things
        time_one = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Openpose deblurring
        if type(im0s) == list:
            myImg = im0s[0]
        else:
            myImg = im0s
        myImg = letterbox(myImg, imgsz, stride=32)[0]
        if not noDeblur:
            myImg = predictor(myImg, None)
        
        # if impossible upper_conf_thres then gets keypoints and if no crops then early exits
        if upper_conf_thres > 1:
            keypoints, humans, Openpose = getKeyPoints(myImg, img, e, pose, noElbow, poseNum)
            cropBoxes = [getCropBoxes(point[0], myImg, outerRatio, device, point[1], Openpose) for point in keypoints]       
            cropBoxes = [torch.Tensor(box).to(device) for box in cropBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
            checkBoxes = [getCropBoxes(point[0], myImg, innerRatio, device, point[1], Openpose) for point in keypoints]
            checkBoxes = [torch.Tensor(box).to(device) for box in checkBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
            
        # running the prediction itself        
        epochs = 1
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        time_two = time_sync()
        for i in range(0, epochs):
            pred = model(img, augment=augment)[0]
        t3 = time_sync()
        dt[1] += t3 - t2
        time_twoHalf = time_sync()
        for i in range(0, epochs):
            bread = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)        
        pred = bread
        time_three = time_sync()
        dt[2] += time_sync() - t3
        
        #print("Dataset Preparation:", time_two - time_one)
        #print("Readying keypoints:", timeDeblurOne - t1)
        #print("Deblurring:", timeDeblurTwo - timeDeblurOne)
        #print("Inference:", (time_twoHalf - time_two) / epochs)
        #print("NMS:", (time_three - time_twoHalf) / epochs)
        #print("Inference + NMS:", (time_three - time_two) / epochs)
        #FLOPs = high.stop_counters()
        
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # checking if any detections have a confidence below a threshold
        upperConfNotExceeded = False
        for i, det in enumerate(pred):
            for detection in det:
                if detection[-2] < upper_conf_thres:
                    upperConfNotExceeded = True
                    break
        
        # creating the crop and checkboxes here if they are needed and not made yet
        if upper_conf_thres <= 1:
            if upperConfNotExceeded:
                keypoints, humans, Openpose = getKeyPoints(myImg, img, e, pose, noElbow, poseNum)
                cropBoxes = [getCropBoxes(point[0], myImg, outerRatio, device, point[1], Openpose) for point in keypoints]       
                cropBoxes = [torch.Tensor(box).to(device) for box in cropBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
                checkBoxes = [getCropBoxes(point[0], myImg, innerRatio, device, point[1], Openpose) for point in keypoints]
                checkBoxes = [torch.Tensor(box).to(device) for box in checkBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]
            else:
                cropBoxes = []
                checkBoxes = []
        
        # new poopy poopies poopster YEAH YEAH YEAAAAAAAAAAAAAAA
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
        
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
        for label in fpLabels:
            label = torch.Tensor(label).to(device)
        for *xyxy, conf, cls in reversed(fpLabels):
            if conf >= conf_thres and (save_img or save_crop or view_img):  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                #plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        ######################################################
        #"""
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

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            # properly scaling the bounding boxes
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
            """   
            if len(det):
                # Rescale boxes from img_size to im0 size and same thing done for crops and check boxes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Check if any overlap between keypoint and checkBoxes (handheld weapon)
                newDet = []
                buttDet = []
                for detection in det:
                    # check if detection is handheld
                    if not handheld:
                        buttDet.append(detection)
                    elif handheld_map.get(int(detection[5])):
                        buttDet.append(detection)
                    # if handheld detection is confident enough then ignores filter
                    if handheld_map.get(int(detection[5])) and detection[-2] >= upper_conf_thres:
                        newDet.append(detection)
                        cv2.putText(im0, "Spider-Sense Tingling!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)                
                    elif outerDet:
                        # checking if overlap between keypoint and cropBoxes
                        for check in checkBoxes:
                            checkOverlap = bbox_iou(detection, check)
                            if (not handheld or handheld_map.get(int(detection[5]))) and checkOverlap > 0:
                                maxWidth = max(detection[3]-detection[1], detection[2]-detection[0])
                                maxCropW = max(check[3]-check[1], check[2]-check[0])
                                if maxWidth/maxCropW <= 2.5:
                                    newDet.append(detection)
                                    cv2.putText(im0, "Spider-Sense Tingling!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)                
                                    break
                    else:
                        # check if detection in out crop
                        for crop in cropBoxes:
                            if (bbox_iou(detection, crop) > 0 and bbox_overlap(detection, crop) >= 0.6):
                                # checking if overlap between keypoint and checkBoxes
                                for check in checkBoxes:
                                    checkOverlap = bbox_iou(detection, check)
                                    if (not handheld or handheld_map.get(int(detection[5]))) and checkOverlap > 0:
                                        newDet.append(detection)
                                        cv2.putText(im0, "Spider-Sense Tingling!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)                
                                        break
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string                
                           
                # Write results (change det if you want to see all boxes and newDet for only valid dets)
                if not allDet:
                    buttDet = newDet
                #print("\n", buttDet)
                for *xyxy, conf, cls in reversed(buttDet):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        #plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            #"""
            
            # write keypoint boxes
            i = 0
            if not outerDet:
                for *xyxy, conf, cls in reversed(cropBoxes):
                    c = int(cls)
                    if xyxy[3] - xyxy[1] > 0 and xyxy[2] - xyxy[0] > 0:
                        #save_one_box(xyxy, imc, file=save_dir/ 'wrist_crops' / names[c] / f'{p.stem}.jpg', BGR=True, pad=0)
                        annotator.box_label(xyxy, keypoints[i][-1], color=colors(c, True))
                    i += 1
            i = 0
            for *xyxy, conf, cls in reversed(checkBoxes):
                c = int(cls)
                if xyxy[3] - xyxy[1] > 0 and xyxy[2] - xyxy[0] > 0:
                    #save_one_box(xyxy, imc, file=save_dir/ 'wrist_crops' / names[c] / f'{p.stem}.jpg', BGR=True, pad=0)
                    annotator.box_label(xyxy, keypoints[i][-1], color=colors(c, True))
                i += 1
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({time_sync() - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                
        # Save results (image with detections)
        if not noPose:
            if Openpose:
                im0 = TfPoseEstimator.draw_humans(im0, humans, imgcopy=False)
            else:
                draw_keypoints(im0, humans, pose.coco_skeletons) 
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

    print(f'Done. ({time_sync() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--det-model', nargs='+', type=str, default='crowdhuman_yolov5m.pt', help='pose detection model path(s)')
    parser.add_argument('--pose-model', nargs='+', type=str, default='./poseEstimation/simdr_hrnet_w48_256x192.pth', help='pose model path(s)')
    parser.add_argument('--weights_path', default=str, help='weight path for DeblurGANv2')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--upper-conf-thres', type=float, default=1.1, help='confidence threshold at which pipeline won\'t be applied')
    parser.add_argument('--pose-conf-thres', type=float, default=0.4, help='confidence threshold for pose detector')
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
    parser.add_argument('--visualize', action='store_true', help='visualize features')
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
    parser.add_argument('--noPose', default=False, action='store_true', help='option for not showing pose')
    parser.add_argument('--allDet', default=False, action='store_true', help='option for showing all detections')
    parser.add_argument('--outerDet', default=False, action='store_true', help='option for using outer det vs outer crop width')
    parser.add_argument('--innerRatio', default=1, type=int, help='inner crop ratio for pipeline') 
    parser.add_argument('--outerRatio', default=2, type=int, help='outer crop ratio for pipeline') 
    parser.add_argument('--poseNum', default=3, type=int, help='number of humans to swtich pose detections') 
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('tensorboard', 'thop'))

    detect(**vars(opt))