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
def getKeyPoints(img, e, noElbows):    
    return
    
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