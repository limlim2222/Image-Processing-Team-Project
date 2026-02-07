import math

import cv2
import numpy as np


def getCars(bgpath, impath):

    # Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_sharpe = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])

    # Import image
    #image = cv2.imread('image3.jpg', cv2.COLOR_BGR2RGB)
    image = cv2.imread(impath, cv2.COLOR_BGR2RGB)
    background = cv2.imread(bgpath, cv2.COLOR_BGR2RGB)

    # Sharpening
    image = cv2.filter2D(image, -1, kernel_sharpe)
    background = cv2.filter2D(background, -1, kernel_sharpe)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    # Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)
    background = cv2.GaussianBlur(background, (5, 5), 0)

    # Background subtraction
    img_sub = cv2.absdiff(image, background)

    # Background subtraction binary
    retval, img_binary = cv2.threshold(img_sub, 45, 255, cv2.THRESH_BINARY)

    # morphology
    opening1 = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    # erosion1 = cv2.erode(opening1,kernel,iterations=1)
    # closing1 = cv2.morphologyEx(erosion1, cv2.MORPH_CLOSE, kernel)
    # closing2 = cv2.morphologyEx(closing1, cv2.MORPH_CLOSE, kernel)
    # dilate1= cv2.dilate(closing2,kernel,iterations=1)
    # erosion2 = cv2.erode(dilate1,kernel,iterations=1)

    # Labeling
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(opening1)
    return cnt, labels, stats, centroids

def compute_intersect_area(rect1, rect2):
    x1, y1 = rect1[0], rect1[1]
    x2, y2 = rect1[0] + rect1[2], rect1[1] + rect1[3]
    x3, y3 = rect2[0], rect2[1]
    x4, y4 = rect2[0] + rect2[2], rect2[1] + rect2[3]

    if x2 < x3:
        return 0

    if x1 > x4:
        return 0

    if y2 < y3:
        return 0

    if y1 > y4:
        return 0

    left_up_x = max(x1, x3)
    left_up_y = max(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = min(y2, y4)

    width = right_down_x - left_up_x
    height = right_down_y - left_up_y

    return (width * height) / (rect1[2] * rect1[3])

bboxes = np.loadtxt('data/20221209_194124_bboxes.csv', delimiter=",")
impath = 'data/test.jpg'
bgpath = 'data/background2.jpg'

cnt, labels, bstats, centroids =  getCars(bgpath, impath)

stats = []

for st in bstats:
    area = st[3] * st[2]
    if(area<150 and area>10):
        stats.append(st)

cnt = len(stats)

n = len(bboxes)

a = [0,0,4,4]
b = [2,3,106,108]
result= compute_intersect_area(b , a)


image1= cv2.imread('data/test.jpg', cv2.COLOR_BGR2RGB)
'''

for i in range(len(stats)):
    cv2.rectangle(image, (int(stats[i][0]), int(stats[i][1])),
              (int(stats[i][0] + stats[i][2]), int(stats[i][1] + stats[i][3])), (255, 0, 255))
'''
image = image1.copy()
for i in range(n):
    a = bboxes[i][:-1]
    cv2.rectangle(image, (int(bboxes[i][0]), int(bboxes[i][1])),
                  (int(bboxes[i][0] + bboxes[i][2]), int(bboxes[i][1] + bboxes[i][3])), (0, 0, 255), 7)

    for j in range(cnt):
        b = stats[j]
        result= compute_intersect_area(a, b)
        #result= compute_intersect_area(bboxes[i] , stats[j])

        if result > 0:
            cv2.rectangle(image, (int(bboxes[i][0]), int(bboxes[i][1])),(int(bboxes[i][0]+bboxes[i][2]), int(bboxes[i][1]+bboxes[i][3])), (255, 255, 255),5)
            #break
        else:
            cv2.rectangle(image, (int(bboxes[i][0]), int(bboxes[i][1])),(int(bboxes[i][0]+bboxes[i][2]), int(bboxes[i][1]+bboxes[i][3])), (0, 0, 0))



        cv2.rectangle(image, (int(stats[j][0]), int(stats[j][1])),
                      (int(stats[j][0] + stats[j][2]), int(stats[j][1] + stats[j][3])), (255, 0, 255))
    cv2.imshow("test", image)
    cv2.waitKey(10)
    a=5
cv2.imshow("test", image)
cv2.waitKey()