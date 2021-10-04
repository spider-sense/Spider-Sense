import math

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
    if w >= h:
        m = w
    else:
        m = h
    humans = e.inference(img, scales=[None])
    
    # Getting keypoints
    KP = []
    for human in humans:
        parts = human.body_parts
        if not (4 in parts or 7 in parts):
            continue
        
        head = human.get_face_box(432, 368)
        if type(head) == dict and head["w"] <= m/8 and parts.get(16) != None and parts.get(0) != None and parts.get(17) != None and max([distGet((parts[16].x * w, parts[16].y * h), (parts[0].x * w, parts[0].y * h)), \
                                                            distGet((parts[0].x * w, parts[0].y * h), (parts[17].x * w, parts[17].y * h))]) < distGet((parts[16].x * w, parts[16].y * h), (parts[17].x * w, parts[17].y * h)):
            headWidth = head["w"]
        else:
            # getting all arm lengths possible
            arms = []
            if 7 in parts and 6 in parts:
                arms.append(distGet((parts[6].x * w, parts[6].y * h), (parts[7].x * w, parts[7].y * h)))
            if 3 in parts and 4 in parts:
                arms.append(distGet((parts[3].x * w, parts[3].y * h), (parts[4].x * w, parts[4].y * h)))
            if 2 in parts and 3 in parts:
                arms.append(distGet((parts[2].x * w, parts[2].y * h), (parts[3].x * w, parts[3].y * h)))
            if 5 in parts and 7 in parts:
                arms.append(distGet((parts[5].x * w, parts[5].y * h), (parts[7].x * w, parts[7].y * h)))
            arms.sort(reverse=True)
            
            # Returning max arm length
            headWidth = False
            for armLength in arms:
                if armLength <= m/8:
                    headWidth = armLength
                    break
            
            # If can't get anything else then just uses upper body bounding box
            if type(headWidth) != False:
                headWidth *= 0.667
            else:
                # moving onto upper bounding box
                upperBody = human.get_upper_body_box(432, 368)
                if type(upperBody) == dict and upperBody["w"] <= m/8:
                    headWidth = upperBody["w"]/2
                else:
                    headWidth = m/16

        # Searching for wrists
        wrists = True
        if 7 in parts:
            KP.append([parts[7], headWidth])
        if 4 in parts:
            KP.append([parts[4], headWidth])
        else:
            wrists = False
        
        # If no wrists then backs up with elbow
        if not wrists:
            if 3 in parts:
                KP.append([parts[3], headWidth])
            if 6 in parts:
                KP.append([parts[6], headWidth]) 
        
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
    
    return int(middleWidth)