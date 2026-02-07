import math
import time
from datetime import datetime

import cv2
import numpy as np

import isect_segments_bentley_ottmann.poly_point_isect as bot

import math

#poly_point_isect
def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)

def findStraightLines(edges):
     ''''''

     lines = cv2.HoughLines(edges  #image
                            , 1#1    #rho distance resolution
                            , np.pi/120#np.pi/180#theta angular resolution
                            , 100 #150#200  #vote threshold
                            , None#200 #srn rho divisor
                            , 0#200   #stn theta
                            , 70#200  #min_theta
                            #, np.pi #max_therta
                            )

     for  r_theta in lines:
         arr = np.array(r_theta[0], dtype =  np.float64)
         r,theta = arr

         a = np.cos(theta)
         b = np.sin(theta)
         x0 = a*r
         y0 = b*r

         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
         cv2.line(im, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

     showInMovedWindow("HoughLines", im, 700,100)


     #cv2.imshow("src", im)
     #cv2.waitKey(0)


def showLines(im, linesP):
    cdstP = np.copy(im)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = np.squeeze(linesP[i])
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)


    return cdstP

def segments_distance(x11, y11, x12, y12, x21, y21, x22, y22):
  """ distance between two segments in the plane:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
  """
  if segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22): return 0
  # try each of the 4 vertices w/the other segment
  distances = []
  distances.append(point_segment_distance(x11, y11, x21, y21, x22, y22))
  distances.append(point_segment_distance(x12, y12, x21, y21, x22, y22))
  distances.append(point_segment_distance(x21, y21, x11, y11, x12, y12))
  distances.append(point_segment_distance(x22, y22, x11, y11, x12, y12))
  return min(distances)

def segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
  """ whether two segments in the plane intersect:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
  """
  dx1 = x12 - x11
  dy1 = y12 - y11
  dx2 = x22 - x21
  dy2 = y22 - y21
  delta = dx2 * dy1 - dy2 * dx1
  if delta == 0: return False  # parallel segments
  s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
  t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta)
  return (0 <= s <= 1) and (0 <= t <= 1)

def point_segment_distance(px, py, x1, y1, x2, y2):
  dx = x2 - x1
  dy = y2 - y1
  if dx == dy == 0:  # the segment's just a point
    return math.hypot(px - x1, py - y1)

  # Calculate the t that minimizes the distance.
  t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

  # See if this represents one of the segment's
  # end points or a point in the middle.
  if t < 0:
    dx = px - x1
    dy = py - y1
  elif t > 1:
    dx = px - x2
    dy = py - y2
  else:
    near_x = x1 + t * dx
    near_y = y1 + t * dy
    dx = px - near_x
    dy = py - near_y

  return math.hypot(dx, dy)

def findStraightLinesPNw(edges):
    linesP = cv2.HoughLinesP(edges #input
                             , 1          # rho distnance resolution
                             , np.pi /180 #180  #angle resolution
                             , 55#45#15#20#50      # threshold : min num of vites
                             ,  np.array([])
                             , 20#25#30#15#4 #8 #50      # min line length
                             , 17 #17#14        # max allowed gap
                             )
    cdstP  =  np.copy(im)


    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)

    #cv2.imshow("Source", cdstP)
    return linesP


def get_lines(lines_in):
    if cv2.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines_in]

def mergeLines(lines):
    # merge lines

    # ------------------
    # prepare
    _lines = []
    for _line in get_lines(lines):
        _lines.append([(_line[0], _line[1]), (_line[2], _line[3])])

    # sort
    _lines_x = []
    _lines_y = []
    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):
            _lines_y.append(line_i)
        else:
            _lines_x.append(line_i)

    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

    merged_lines_x = merge_lines_pipeline_2(_lines_x)
    merged_lines_y = merge_lines_pipeline_2(_lines_y)

    merged_lines_all = []
    merged_lines_all.extend(merged_lines_x)
    merged_lines_all.extend(merged_lines_y)
    print("process groups lines", len(_lines), len(merged_lines_all))

    '''reformed'''
    rlines = []

    for ml in merged_lines_all:
        rline = [ml[0][0], ml[0][1], ml[1][0], ml[1][1]]
        rlines.append(rline)

    #img_merged_lines = mpimg.imread(image_src)
    #return merged_lines_all
    return rlines

def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1],
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1],
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1],
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1],
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])

    return min(dist1, dist2, dist3, dist4)

def lineMagnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude

# Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
# https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
# http://paulbourke.net/geometry/pointlineplane/
def DistancePointLine(px, py, x1, y1, x2, y2):
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        # // closest point does not fall within the line segment, take the shorter distance
        # // to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine

def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 30
    min_angle_to_merge = 30

    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                    orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                    if int(abs(
                            abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        # print("angles", orientation_i, orientation_j)
                        # print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                    orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                    if int(abs(
                            abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        # print("angles", orientation_i, orientation_j)
                        # print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        # lines[idx] = False
            # append new group
            super_lines.append(new_group)

    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))

    return super_lines_final

def merge_lines_segments1(lines, use_log=False):
    if (len(lines) == 1):
        return lines[0]

    #lines = np.zeros((len(olines), 2, 2))
    #a = olines[:, 0,:2]

    #lines[:, 0] = olines[:, 0,:2]
    #lines[:, 1] = olines[:, 0,2:]
    line_i = lines[0]

    # orientation
    orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))

    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])

    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):

        # sort by y
        points = sorted(points, key=lambda point: point[1])

        if use_log:
            print("use y")
    else:

        # sort by x
        points = sorted(points, key=lambda point: point[0])

        if use_log:
            print("use x")

    return [points[0], points[len(points) - 1]]



def gradients(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    Ix = cv2.filter2D(im, -1, Kx)
    Iy = cv2.filter2D(im, -1, Ky)

    #cv2.imshow('x', Ix)
    #cv2.imshow('y', Iy)
    #cv2.waitKey(0)
    return Ix, Iy

def combineGrads(X, Y):
    c  = np.where(X>Y, X, Y)
    #c = np.where(c<50, 0, c )
    c = np.where(c<50, 0, c )
    #c  = np.where(X>10 or Y>10, 255, 0)
    return c

def thresh(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(im, 110, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binary Threshold', thresh1)
    cv2.waitKey(0)

def findIntersections(lines):
    ''''''

    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            #cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    #lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    #print(lines_edges.shape)
    #cv2.imwrite('line_parking.png', lines_edges)

    intersections = bot.isect_segments(points)

    #polygons =  bot.isect_polygon(points[:5])

    for idx, inter in enumerate(intersections):
        a, b = inter
        match = 0
        for other_inter in intersections[idx:]:
            if other_inter == inter:
                continue
            c, d = other_inter
            if abs(c - a) < 15 and abs(d - b) < 15:
                match = 1
                intersections[idx] = ((c + a) / 2, (d + b) / 2)
                intersections.remove(other_inter)

        if match == 0:
            intersections.remove(inter)

    return intersections

def showIntersections2(im_in, intersections):
    lines_edges  =  im_in.copy()
    for inter in intersections:
        a, b = inter
        for i in range(3):
            for j in range(3):
                lines_edges[int(b) + i, int(a) + j] = [0, 255, 0]

    return lines_edges

def showIntersections(im_in, intersections):
    im_out = im_in.copy()

    for pt in intersections:
            im_out = cv2.circle(im_out, (int(pt[0]), int(pt[1])), radius=3
                                , color=(0, 0, 255), thickness=-1)

    return im_out

def lineHas(line, inter):
    a =  inter
    b = [line[0][0],line[0][1]]
    c = [line[0][2],line[0][3]]
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    epsilon = 1
    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True

def lineHasOriginal(line, inter):
    b, c = line
    a =  inter
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    epsilon = 1
    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True


def getCTime():
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    return date_time

def genBBoxes(im, lines, intersections):
    #lines2  = sorted(lines)

    #im = np.zeros((600, 1200,3))

    bboxes = []
    n = len(intersections)
    im2 = im.copy()
    a,b,c,d,e,f = 90, 170, 277 , 420, 473, 610
    rows = [90, 170, 277 , 420, 473, 620]
    r = [20, 30, 40 , 40, 40, 30]
    '''
    im2 = cv2.line(im2, (0, a), (900, a), color=(255, 0, 0), thickness=1)
    im2 = cv2.line(im2, (0, b), (900, b), color=(0, 255, 0), thickness=1)
    im2 = cv2.line(im2, (0, c), (900, c), color=(0, 0, 255), thickness=1)
    im2 = cv2.line(im2, (0, d), (900, d), color=(255, 0, 255), thickness=1)
    im2 = cv2.line(im2, (0, e), (900, e), color=(0, 0, 255), thickness=1)
    im2 = cv2.line(im2, (0, f), (900, f), color=(255, 0, 10), thickness=1)'''

    '''correct intersections'''
    inters2 = []
    for inter in intersections:
        nearest_row = min(rows, key=lambda x: abs(x - inter[1]))
        ninter = [inter[0], nearest_row]
        inters2.append(ninter)

    inters2  = sorted(inters2, key=lambda k: [k[1], k[0]])

    cn = np.random.randint(2, size=(n, 3)).tolist()
    cn = [[255,0,0], [0, 255, 0], [0, 255, 0], [255, 255, 0], [0, 255, 255]]
    bboxes = []
    for i in range(5,n-1):
        inter = inters2[i]

        nr= rows.index(min(rows, key=lambda x: abs(x - inter[1])))
        st = (int(inter[0]), int(inter[1])-r[nr])
        #h  = inter[1] + r*2
        #w  = inters2[i+1][0] - inters2[i][0]

        h  = r[nr]*2
        w  = inters2[i+1][0] - inters2[i][0]
        en = int(st[0]+w) , int(st[1]+h)


        if( w> 50 or w<10):
            continue

        bbox = [st[0], st[1],  w, h]
        bbox1 = [round(st[0], 4), round(st[1], 4), w , round(h/2,4)]
        bbox2 = [round(st[0], 4), round(st[1]+h/2,4), w, round(h/2,4)]
        ''' '''
        if(nr <= 2):
        #bboxes.append(bbox)
            bboxes.append(bbox1)
            bboxes.append(bbox2)

        elif(nr == 3):
            bboxes.append(bbox1)

        elif(nr == 4):
            bboxes.append(bbox2)

        elif (nr == 5):
            bboxes.append(bbox1)


    a = len(bboxes)
    bbs = []
    for i in range(a):

        bbox = bboxes[i]
        bbox.append(i+1)
        bbs.append(bbox)
        st = [int(bbox[0]), int(bbox[1])]
        w,h = bbox[2], bbox[3]
        en = int(st[0] + w), int(st[1] + h)
        #en = int(st[0] + w , int(st[1] + h)
        im2 = cv2.rectangle(im2
                            , st
                            , en
                            , color=cn[i%5]#(200, 50, 100)
                            , thickness=2)

        showInMovedWindow("BBoxes", im2, 1200, 100)
        cv2.waitKey(100)

    bb = np.array(bbs)
    svpath = "data/{}_bboxes.csv".format(getCTime())
    np.savetxt(svpath, bb, delimiter=",")
    #cv2.waitKey(0)



    n = len(inters2)
    cn = np.random.randint(2, size=(n, 3)).tolist()



def blur(im):
    dst = cv2.GaussianBlur(im, (3,3), 0)
    #dst = cv2.medianBlur(im, 5)
    #kernel = np.ones((5, 5), np.float32) / 25
    #dst = cv2.filter2D(im, -1, kernel)
    return dst

def applyOpening(im):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    return opening

def findGreen(im):
    aim = im.copy()
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = np.array([0, 0, 0])
    brown_hi = np.array([100, 100, 100])
    #brown_hi = np.array([150, 150, 150])

    # Mask image to only select browns
    mask = cv2.inRange(hsv, brown_lo, brown_hi)

    # Change image to red where we found brown
    aim[mask > 0] = (0, 0, 255)

    showInMovedWindow("Background", aim, 1200, 400)
    #cv2.imshow("test", aim)

    return mask

def findGray(im):
    aim = im.copy()
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = 0
    brown_hi = 155

    # Mask image to only select browns
    mask = cv2.inRange(hsv, brown_lo, brown_hi)

    # Change image to red where we found brown
    aim[mask > 0] = (0, 0, 255)

    showInMovedWindow("Background", aim, 1200, 400)
    #cv2.imshow("test", aim)

    return mask

def kmeansSeg(sample_image):
    #sample_image = cv2.imread('image.jpg')
    img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    #attempts = 120
    attempts = 10

    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return  result_image, center

def substractMask(im, mask):
    im[mask > 0] = 0#(0, 0, 255)
    return im

def findLines(im):
    ''''''
    seg_masks, centers = kmeansSeg(im)

    X, Y = gradients(seg_masks)
    G = combineGrads(X, Y)
    #G[G<120] = 0
    G[G<101] = 0

    linesDetected = findStraightLinesPNw(G)
    #linesDetected = mergeLines(linesDetected)
    imLines = showLines(im, linesDetected)
    intersections = findIntersections(linesDetected)
    imIntersections = showIntersections2(imLines, intersections)

    showInMovedWindow("KMeans Segmentation", seg_masks, 100, 100)
    showInMovedWindow("Gradients", G, 500, 100)
    #showInMovedWindow("Hough Transform P", imLines, 600, 100)
    showInMovedWindow("Bentley Ottman Intersection", imIntersections, 800, 100)

    genBBoxes(im, linesDetected, intersections)

    cv2.waitKey(0)



def testBboxes(im):

    bboxes = np.genfromtxt('data/20221208_170727_bboxes.csv', delimiter=',')
    n = len(bboxes)
    for i in range(n):
        bbox = bboxes[i]

        st = [int(bbox[0]), int(bbox[1])]
        h, w = bbox[2], bbox[3]
        en = int(st[0] + w), int(st[1] + h)

        im = cv2.rectangle(im
                           , st
                           , en
                           , color=(200, 50, 100)
                           , thickness=2)

        cv2.imshow("BBoxes", im)
    cv2.waitKey(0)


impath = "test_images/background4.jpg"
im = cv2.imread(impath)
#im = blur(im)
findLines(im)
