import math
from utils.general import non_max_suppression
from pose_estimation.pose.utils.boxes import letterbox, scale_boxes, non_max_suppression, xyxy2xywh
from pose_estimation.pose.utils.decode import get_final_preds, get_simdr_final_preds

"""
basic function to get distance between two points
"""
def distGet(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

"""
This is the function I made to get the keypoints.  I'll be taking a crop at
each of these keypoints and feeding it to the actual object detector.  The 
keypoints will preferably be the wrists but if no wrists are detected it will
settle for the elbows.  Switches between two pose detectors based on conditions.
img is array and img1 is tensor
"""
def getKeyPoints(img, img1, e, pose, noElbows, poseNum):    
    if poseNum == 0:
        # Running inference
        w = img.shape[1]
        h = img.shape[0]
        if w >= h:
            m = w
        else:
            m = h
        humans = e.inference(img, scales=[None])
        
        # Getting keypoints
        KP = []
        cropWidth = -1
        imType = ""
        upperLim = 6
        for human in humans:
            parts = human.body_parts
            #if not (parts.get(4) or parts.get(7)):
            if not (parts.get(4) or parts.get(7) or (not noElbows and parts.get(3)) or (not noElbows and parts.get(6))):
                continue
            
            head = human.get_face_box(w, h)
            if type(head) == dict and head["w"] <= m/upperLim and parts.get(16) != None and parts.get(0) != None and parts.get(17) != None and max([distGet((parts[16].x * w, parts[16].y * h), (parts[0].x * w, parts[0].y * h)), \
                                                                distGet((parts[0].x * w, parts[0].y * h), (parts[17].x * w, parts[17].y * h))]) < distGet((parts[16].x * w, parts[16].y * h), (parts[17].x * w, parts[17].y * h)):
                cropWidth = head["w"]
                imType = "h"
            else:
                # getting all arm lengths possible
                arms = []
                if parts.get(7) and parts.get(6):
                    arms.append(distGet((parts[6].x * w, parts[6].y * h), (parts[7].x * w, parts[7].y * h)))
                if parts.get(3) and parts.get(4):
                    arms.append(distGet((parts[3].x * w, parts[3].y * h), (parts[4].x * w, parts[4].y * h)))
                if parts.get(2) and parts.get(3):
                    arms.append(distGet((parts[2].x * w, parts[2].y * h), (parts[3].x * w, parts[3].y * h)))
                if parts.get(5) and parts.get(7):
                    arms.append(distGet((parts[5].x * w, parts[5].y * h), (parts[7].x * w, parts[7].y * h)))
                arms.sort(reverse=True)
                
                # Returning max arm length
                for armLength in arms:
                    if armLength * 0.667 <= m/upperLim:
                        cropWidth = armLength
                        imType = "a"
                        break
                
                # If can't get anything else then just uses upper body bounding box
                if cropWidth > 0:
                    cropWidth *= 0.667
                else:
                    # moving onto upper bounding box
                    upperBody = human.get_upper_body_box(w, h)
                    if type(upperBody) == dict and (upperBody["w"] / 2) <= m/upperLim:
                        cropWidth = upperBody["w"]/2
                        imType = "u"
                    else:
                        cropWidth = m/upperLim
                        imType = "f"
                        
             # Searching for wrists and settling for elbows if needed and allowed
            if parts.get(7):
                KP.append([parts[7], cropWidth, imType])
            elif not noElbows and parts.get(6):
                KP.append([parts[6], cropWidth, imType])
            if parts.get(4):
                KP.append([parts[4], cropWidth, imType])
            elif not noElbows and parts.get(3):
                KP.append([parts[3], cropWidth, imType])
            
        return KP, humans, True
    elif poseNum == -1:
        # getting necessary predictions
        pred = pose.det_model(img1)[0]
        pred = non_max_suppression(pred, pose.conf_thres, pose.iou_thres, classes=0)
        
        # getting necessary poses
        coords = None
        for det in pred:
            if len(det):
                boxes = scale_boxes(det[:, :4], img.shape[:2], img1.shape[-2:]).cpu()
                boxes = pose.box_to_center_scale(boxes)
                outputs = pose.predict_poses(boxes, img)
                
                if 'simdr' in pose.model_name:
                    coords = get_simdr_final_preds(*outputs, boxes, pose.patch_size)
                else:
                    coords = get_final_preds(outputs, boxes)
        humans = coords
        if humans is None:
            return [], [], False
        
        # getting all keypoints
        KP = []
        cropWidth = -1
        imType = 'h'
        for human in humans:
            headWidth = distGet(human[3], human[4])
            if headWidth > max([distGet(human[0], human[3]), distGet(human[0], human[4])]):
                cropWidth = headWidth
                imType = 'h'
            else:
                arms = []
                arms.append(distGet(human[6], human[8]))
                arms.append(distGet(human[8], human[10]))
                arms.append(distGet(human[5], human[7]))
                arms.append(distGet(human[7], human[9]))
                arms.sort(reverse=True)
                cropWidth = 0.667 * arms[0]
                imType = 'a'
            KP.append([human[9], cropWidth, imType])
            KP.append([human[10], cropWidth, imType])
            
        return KP, humans, False
    else:
        # getting necessary predictions
        pred = pose.det_model(img1)[0]
        pred = non_max_suppression(pred, pose.conf_thres, pose.iou_thres, classes=0)
        coords = None
        for det in pred:
            # if no humans detected then early exit
            if len(det) == 0:
                return [], [], False
            if len(det) <= poseNum:
                boxes = scale_boxes(det[:, :4], img.shape[:2], img1.shape[-2:]).cpu()
                boxes = pose.box_to_center_scale(boxes)
                outputs = pose.predict_poses(boxes, img)
                
                if 'simdr' in pose.model_name:
                    coords = get_simdr_final_preds(*outputs, boxes, pose.patch_size)
                else:
                    coords = get_final_preds(outputs, boxes)


        # if coords are found then uses them else uses Openpose
        if not (coords is None):            
            # getting all keypoints
            humans = coords
            KP = []
            cropWidth = -1
            imType = 'h'
            for human in humans:
                headWidth = distGet(human[3], human[4]) 
                if headWidth > max([distGet(human[0], human[3]), distGet(human[0], human[4])]):
                    cropWidth = headWidth
                    imType = 'h'
                else:
                    arms = []
                    arms.append(distGet(human[6], human[8]))
                    arms.append(distGet(human[8], human[10]))
                    arms.append(distGet(human[5], human[7]))
                    arms.append(distGet(human[7], human[9]))
                    arms.sort(reverse=True)
                    cropWidth = 0.667 * arms[0]
                    imType = 'a'
                KP.append([human[9], cropWidth, imType])
                KP.append([human[10], cropWidth, imType])
                
            return KP, humans, False
        else:
            # Running inference
            w = img.shape[1]
            h = img.shape[0]
            if w >= h:
                m = w
            else:
                m = h
            humans = e.inference(img, scales=[None])
            
            # Getting keypoints
            KP = []
            cropWidth = -1
            imType = ""
            upperLim = 6
            for human in humans:
                parts = human.body_parts
                #if not (parts.get(4) or parts.get(7)):
                if not (parts.get(4) or parts.get(7) or (not noElbows and parts.get(3)) or (not noElbows and parts.get(6))):
                    continue
                
                head = human.get_face_box(w, h)
                if type(head) == dict and head["w"] <= m/upperLim and parts.get(16) != None and parts.get(0) != None and parts.get(17) != None and max([distGet((parts[16].x * w, parts[16].y * h), (parts[0].x * w, parts[0].y * h)), \
                                                                    distGet((parts[0].x * w, parts[0].y * h), (parts[17].x * w, parts[17].y * h))]) < distGet((parts[16].x * w, parts[16].y * h), (parts[17].x * w, parts[17].y * h)):
                    cropWidth = head["w"]
                    imType = "h"
                else:
                    # getting all arm lengths possible
                    arms = []
                    if parts.get(7) and parts.get(6):
                        arms.append(distGet((parts[6].x * w, parts[6].y * h), (parts[7].x * w, parts[7].y * h)))
                    if parts.get(3) and parts.get(4):
                        arms.append(distGet((parts[3].x * w, parts[3].y * h), (parts[4].x * w, parts[4].y * h)))
                    if parts.get(2) and parts.get(3):
                        arms.append(distGet((parts[2].x * w, parts[2].y * h), (parts[3].x * w, parts[3].y * h)))
                    if parts.get(5) and parts.get(7):
                        arms.append(distGet((parts[5].x * w, parts[5].y * h), (parts[7].x * w, parts[7].y * h)))
                    arms.sort(reverse=True)
                    
                    # Returning max arm length
                    for armLength in arms:
                        if armLength * 0.667 <= m/upperLim:
                            cropWidth = armLength
                            imType = "a"
                            break
                    
                    # If can't get anything else then just uses upper body bounding box
                    if cropWidth > 0:
                        cropWidth *= 0.667
                    else:
                        # moving onto upper bounding box
                        upperBody = human.get_upper_body_box(w, h)
                        if type(upperBody) == dict and (upperBody["w"] / 2) <= m/upperLim:
                            cropWidth = upperBody["w"]/2
                            imType = "u"
                        else:
                            cropWidth = m/upperLim
                            imType = "f"
                            
                 # Searching for wrists and settling for elbows if needed and allowed
                if parts.get(7):
                    KP.append([parts[7], cropWidth, imType])
                elif not noElbows and parts.get(6):
                    KP.append([parts[6], cropWidth, imType])
                if parts.get(4):
                    KP.append([parts[4], cropWidth, imType])
                elif not noElbows and parts.get(3):
                    KP.append([parts[3], cropWidth, imType])
                
            return KP, humans, True
            
"""
Function for getting the crop of an image.  Will look at headWidth of img.
"""
def getCropBoxes(point, img, factor, device, cropWidth, Openpose):
    if Openpose:
        cropWidth *= factor
        pointX = round(img.shape[1] * point.x)
        pointY = round(img.shape[0] * point.y)
        lowX = pointX - cropWidth
        upX = pointX + cropWidth
        lowY = pointY - cropWidth
        upY = pointY + cropWidth
        box = [lowX, lowY, upX, upY, 0, 0]
        return box
    else:
        cropWidth *= factor
        pointX = point[0]
        pointY = point[1]
        lowX = pointX - cropWidth
        upX = pointX + cropWidth
        lowY = pointY - cropWidth
        upY = pointY + cropWidth
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
    
    return int(middleWidth)