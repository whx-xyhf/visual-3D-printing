from tkinter import Toplevel
import cv2 as cv
import numpy as np
import pylab
import base64
from io import BytesIO
import matplotlib
from scipy.signal import savgol_filter
import scipy.signal as signal
from collections import Counter
import itertools
matplotlib.use('Agg')


def sectionContourDraw(x, y, fitting_strength):
    if x == [] or y == []:
        return ''
    Factor = np.polyfit(y, x, fitting_strength)
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'black', label='')
    return Factor


def functionFitting(x, y):
    '''
    函数拟合
    输入：(x, y)分别为拟合函数的坐标轴数据
    输出：在原画布上添加该函数线段
    '''
    if x == [] or y == []:
        return ''
    Factor = np.polyfit(y, x, 8)
    drawFunction(Factor, y)
    return Factor


def drawFunction(Factor, y):
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'black', label='')


def pointTransform(image_width, image_height, real_width, real_height, x, y):
    real_x = real_width / image_width * x
    real_y = real_height / image_height * y
    return [real_x, real_y]


def relativeSlope(inputList):
    '''
    # 相对斜率
    输入为列表
    返回为数组
    '''
    recentChange = 0
    relativeSlopeList = []
    for i in range(len(inputList)):
        if (recentChange-inputList[i])/inputList[i] < 0.03 and (recentChange-inputList[i])/inputList[i] > -0.03:
            relativeSlopeList.append(0)
            recentChange = inputList[i]
            continue
        relativeSlopeList.append((recentChange-inputList[i])/inputList[i])
        recentChange = inputList[i]
    return np.array(relativeSlopeList)


def getTopLimit(imgMat, threshold):
    '''
    上边界判定
    输入图像矩阵
    输出上边界y值
    '''
    if threshold < 200:
        xy = list(np.where(imgMat < 128))
    else:
        xy = list(np.where(imgMat == 255))
    xy = splitArray(xy[0], xy[1])
    leftContour = [[], []]
    rightContour = [[], []]
    pixelStatistics = []

    for i in range(len(xy[1])):
        leftContour[0].append(xy[0][i][0])
        rightContour[0].append(xy[0][i][-1])
        pixelStatistics.append(xy[0][i][-1]-xy[0][i][0])

    # pixelStatistics = savgol_filter(pixelStatistics, 5, 3)
    der1Contour = relativeSlope(pixelStatistics)
    # # 首批不为零元素的最后一个
    # for i in range(len(der1Contour)-1):
    #     if der1Contour[i] > 0 and der1Contour[i+1] == 0:
    #         return i
    topLimit = np.array(signal.argrelextrema(der1Contour, np.greater)[0])
    return topLimit[0]


def getFinalContour(image_buffer,  fitting_strength):
    img_org = cv.imdecode(np.frombuffer(
        image_buffer, np.uint8), cv.IMREAD_GRAYSCALE)
    image_width = img_org.shape[1]
    image_height = img_org.shape[0]
    img_org = cv.bitwise_not(img_org)
    ret, img_bin = cv.threshold(img_org, 128, 255, cv.THRESH_TRIANGLE)

    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv.erode(img_bin, kernel, iterations=1)
    img_bin = cv.dilate(img_bin, kernel, iterations=1)

    img_thinning = cv.ximgproc.thinning(
        img_bin, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
    img_thinning = cv.ximgproc.thinning(img_org)
    midLine = [[], []]
    midLineXY = np.where(img_thinning == 255)
    midLine[0] = midLineXY[1]
    midLine[1] = midLineXY[0]

    leftContour = [[], []]
    rightContour = [[], []]
    # 每行像素统计
    pixelStatistics = []
    # 记录图像宽度一阶变化率
    der1Contour = []
    # 原图像已取反，故原区域现为白色
    img = np.array(img_org)
    xy = list(np.where(img > 128))
    imgData = [[], []]
    imgData[0] = xy[1]
    imgData[1] = xy[0]
    xy = splitArray(xy[0], xy[1])
    leftContour = [[], []]
    rightContour = [[], []]
    for i in range(len(xy[1])):
        leftContour[0].append(xy[0][i][0])
        leftContour[1].append(xy[1][i][0])
        rightContour[0].append(xy[0][i][-1])
        rightContour[1].append(xy[1][i][-1])
        pixelStatistics.append(xy[0][i][-1]-xy[0][i][0])

    pylab.figure(figsize=(16, 9))

    der1Contour = np.diff(pixelStatistics)
    topLimit = getTopLimit(img, 255)

    # 左轮廓边界为最后的极小值
    leftContourLimit = np.array(
        signal.argrelextrema(der1Contour, np.less)[0])[-1]

    # 右轮廓边界为首个极大值
    for i in signal.argrelextrema(der1Contour, np.greater)[0]:
        if der1Contour[i] > 150:
            rightContourLimit = i
            break

    rightContour[0] = rightContour[0][topLimit:rightContourLimit]
    rightContour[1] = rightContour[1][topLimit:rightContourLimit]
    leftContour[0] = leftContour[0][topLimit:leftContourLimit]
    leftContour[1] = leftContour[1][topLimit:leftContourLimit]

    der1midLine = np.diff(midLine[0])
    for i in signal.argrelextrema(der1midLine, np.greater)[0]:
        if der1midLine[i] > 150:
            midLine[0] = midLine[0][:i]
            midLine[1] = midLine[1][:i]
            break
    for i in range(len(midLine[1])):
        if midLine[1][i] >= topLimit:
            midLine[0] = midLine[0][i:]
            midLine[1] = midLine[1][i:]
            break

    # pylab.plot(image[0], image[1], 'w')
    fy1 = sectionContourDraw(leftContour[0], leftContour[1], fitting_strength)
    fy2 = sectionContourDraw(
        rightContour[0], rightContour[1], fitting_strength)
    fy3 = sectionContourDraw(midLine[0], midLine[1], fitting_strength)
    # print('f',fy1, fy2, fy3)
    message = ''
    if fy1 == '':
        message += 'leftContour is wrong'+'\n'
    if fy2 == '':
        message += 'rightContour is wrong'+'\n'
    if fy3 == '':
        message += 'midLine is wrong'+'\n'
    pylab.ylim(image_height, 0)
    pylab.xlim(leftContourLimit - 10, image_width - rightContourLimit - 10)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    sio = BytesIO()
    # pylab.show()
    pylab.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = str(data)
    # # 记得关闭，不然画出来的图是重复的
    pylab.close()
    value = sio.getvalue()
    sio.close()
    return fy1, fy2, fy3, topLimit, leftContourLimit, rightContourLimit, src, value


def splitArray(inputYArray, inputXArray):
    returnArray = [[], []]
    n = 0
    for i in range(len(inputYArray)-1):
        # print(inputYArray[i] ,inputYArray[i+1])
        if inputYArray[i] != inputYArray[i+1]:
            returnArray[0].append(inputXArray[n:i+1])
            returnArray[1].append(inputYArray[n:i+1])
            n = i+1
    returnArray.append(inputYArray[n:])
    return returnArray


# 拟合评估
def fittingAssessment(inputArray1, inputArray2):
    return abs_sum(list(np.array(inputArray1)-np.array(inputArray2)))


# 列表元素绝对值之和
def abs_sum(L):
    if L == []:
        return 0
    return abs_sum(L[1:]) + abs(L[0])


def getBottomLineByColumn(imgMat):
    img = imgMat
    xy = list(np.where(img.T <= 1))
    xy = splitArray(xy[0], xy[1])
    leftContour = [[], []]
    rightContour = [[], []]
    pixelStatistics = []

    for i in range(len(xy[1])):
        leftContour[0].append(xy[0][i][0])
        rightContour[0].append(xy[0][i][-1])
        pixelStatistics.append(xy[0][i][-1]-xy[0][i][0])

    pixelStatistics = savgol_filter(pixelStatistics, 5, 3)
    der1Contour = np.diff(pixelStatistics)
    leftFirstMutation = np.array(
        signal.argrelextrema(der1Contour, np.greater)[0])[0]
    leftContourLimit = list(
        Counter(leftContour[0][:leftFirstMutation]).keys())[0]
    return leftContourLimit


def imgPatch(img, leftC, rightC, topLimit):
    """
    上部缺口修补
    """
    offset = 10
    for row in range(topLimit-offset, topLimit+offset):
        if leftC[0][row] > max(min(leftC[0][row-offset:row]), min(leftC[0][row+1:row+offset])):
            img[0][row] = np.append(img[0][row], max(
                min(leftC[0][row-offset:row]), min(leftC[0][row+1:row+offset])))
            img[1][row] = np.append(img[1][row], row)
        if rightC[0][row] < min(max(rightC[0][row-offset:row]), max(rightC[0][row+1:row+offset])):
            img[0][row] = np.append(img[0][row], min(
                max(rightC[0][row-offset:row]), max(rightC[0][row+1:row+offset])))
            img[1][row] = np.append(img[1][row], row)


def getSilhouette(image_buffer, low_Threshold=50, height_Threshold=150, kernel_size=3):
    img = cv.imdecode(np.frombuffer(image_buffer, np.uint8), cv.IMREAD_COLOR)
    # img = cv.imread(image_buffer)
    image_width = img.shape[1]
    image_height = img.shape[0]
    new_grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    detected_edges = cv.GaussianBlur(new_grayImage, (3, 3), 0)
    detected_edges = cv.Canny(detected_edges,
                              low_Threshold,
                              height_Threshold,
                              apertureSize=kernel_size)

    Contour = cv.findContours(
        detected_edges,
        1,
        1,
    )
    contours = Contour[0]
    imageCountour = np.ones(detected_edges.shape, np.uint8)*255
    cv.drawContours(imageCountour, contours, -1, (0, 255, 0), 1)

    img = np.array(imageCountour)
    xy = list(np.where(img < 128))
    xy = splitArray(xy[0], xy[1])
    leftContour = [[], []]
    rightContour = [[], []]
    for i in range(len(xy[1])):
        leftContour[0].append(xy[0][i][0])
        leftContour[1].append(xy[1][i][0])
        rightContour[0].append(xy[0][i][-1])
        rightContour[1].append(xy[1][i][-1])
    topLimit = getTopLimit(img, 128)
    print(topLimit)
    imgPatch(xy, leftContour, rightContour, topLimit)

    leftContourLimit = getBottomLineByColumn(img)
    fittingAssess = 0
    fittingAssessList = []
    for i in range(int(img.shape[0]/100)):
        y = leftContour[1][:leftContourLimit - i]
        x = leftContour[0][:leftContourLimit-i]
        Factor = np.polyfit(y, x, 14)
        F = np.poly1d(Factor)
        fX = F(y)
        fittingAssessList.append(fittingAssess/fittingAssessment(fX, x))
        fittingAssess = fittingAssessment(fX, x)
    leftContourLimit -= fittingAssessList.index(max(fittingAssessList))
    leftContour[0] = leftContour[0][:leftContourLimit]
    leftContour[1] = leftContour[1][:leftContourLimit]
    for i in range(len(xy[1])):
        if xy[1][i][0] >= leftContourLimit:
            xy[0] = xy[0][:i]
            xy[1] = xy[1][:i]
            break
    imgData = [[], []]
    imgData[0] = list(itertools.chain.from_iterable(xy[0]))
    imgData[1] = list(itertools.chain.from_iterable(xy[1]))
    pylab.figure(figsize=(16, 9))
    pylab.plot(imgData[0],  imgData[1], 'black')
    pylab.ylim(image_height, 0)
    pylab.xlim(0, image_width)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    # pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.margins(0.0)
    sio = BytesIO()
    pylab.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = str(data)
    # # 记得关闭，不然画出来的图是重复的
    pylab.close()
    value = sio.getvalue()
    sio.close()
    return src, value


def normalLine(factor, y):
    f = np.poly1d(factor)
    x = f(y)
    derF = f.deriv(1)
    derX = derF(y)
    ky = -1/derX
    const = x-ky*y
    lineFactor = np.array([ky, const])
    return lineFactor, x


def getDistance(p1, p2):
    return (((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5).item()


def strToNdarray(string):
    string = string[1:-1]
    strArray = string.split(' ')
    numA = []
    for i in range(len(strArray)):
        if strArray[i] in ['']:
            continue
        if strArray[i][-2:] in ['\n']:
            strArray[i] = strArray[i][:-3]
        numA.append(float(strArray[i]))
    arry = np.array(numA)
    return arry


def dataExport(fy1, fy2, midLineFactor, yList, bottom, top):
    # fy1 = strToNdarray(fy1)
    # fy2 = strToNdarray(fy2)
    # midLineFactor = strToNdarray(midLineFactor)
    # yList = np.array([600.0, 550.0, 500.0, 450.0, 400.0])
    rList = []
    for y in yList:
        dis1 = -1.0
        dis2 = -1.0
        dis = [[], []]
        normalL, x = normalLine(midLineFactor, y)
        leftLineF = fy1.copy()
        rightLineF = fy2.copy()
        p = getIntersection(leftLineF, normalL, bottom, top)
        dis1 = getDistance([x, y], p)
        p = getIntersection(rightLineF, normalL, bottom, top)
        dis2 = getDistance([x, y], p)
        if(dis1 > 0 and dis2 > 0):
            dis[0].append(dis1)
            dis[1].append(dis2)
        rList.append(dis)
    return rList


def getIntersection(factor1, factor2, bottomLim, topLim):
    '''
    交点计算
    输入：\n
          factor1：函数1的因数\n
          factor2：函数2的因数\n
          bottomLim：下边界限制\n
          topLim：上边界限制\n

          tips:上边界默认大于下边界，函数内会自动调整
    输出：返回两个函数的交点坐标
    '''
    if bottomLim > topLim:
        bottomLim, topLim = topLim, bottomLim
    subDis = len(factor1)-len(factor2)
    for i in range(len(factor2)):
        factor1[i+subDis] -= factor2[i]
    p1 = np.poly1d(factor1)
    p2 = np.poly1d(factor2)
    y = 0
    for i in range(len(p1.roots)):
        if np.imag(p1.roots[i]) == 0 and np.real(p1.roots[i]) < topLim and np.real(p1.roots[i]) > bottomLim:
            y = np.real(p1.roots[i])
    x = p2(y)
    return [x, y]


def factorToPoly(Factor):
    string = ''
    for i in range(len(Factor)):
        if (str(Factor[i]))[0] != '-' and i != 0:
            string += '+'
        string += str(Factor[i])+'*y**'+str(len(Factor)-i-1)
    string += '-x'
    return string


def numList(p1, p2):
    if p1 < p2:
        return (range(int(p1), int(p2)))
    else:
        return (range(int(p2), int(p1)))


def drawNormalLine(yFactor, x):
    xFactor = np.array([1/yFactor[0], -yFactor[1]/yFactor[0]])
    # print(yFactor, xFactor)
    F = np.poly1d(xFactor)
    fY = F(x)
    pylab.plot(x, fY,  'red', label='')


def drawRadiusPic(count, image_ori_width, image_ori_height, fy1, fy2, midLineFactor, topLimit, leftContourLimit, rightContourLimit):
    bottom = rightContourLimit
    leftLineF = strToNdarray(fy1)
    rightLineF = strToNdarray(fy2)
    midLineFactor = strToNdarray(midLineFactor)
    pylab.figure(figsize=(16, 9))
    drawFunction(leftLineF, numList(topLimit, leftContourLimit))
    drawFunction(rightLineF, numList(topLimit, rightContourLimit))
    drawFunction(midLineFactor, numList(topLimit, rightContourLimit))
    yList = np.linspace(topLimit, bottom, count+1, endpoint=False)[1:]
    for y in yList:
        normalLineF, x = normalLine(midLineFactor, y)
        xRange = numList((getIntersection(leftLineF.copy(), normalLineF, bottom, topLimit)[0]),
                         (getIntersection(rightLineF.copy(), normalLineF, bottom, topLimit)[0]))
        drawNormalLine(normalLineF, xRange)
    pylab.ylim(image_ori_height, 0)
    pylab.xlim(0, image_ori_width)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    pylab.margins(0.0)
    sio = BytesIO()
    pylab.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = str(data)
    # # 记得关闭，不然画出来的图是重复的
    pylab.close()
    sio.close()
    rList = dataExport(leftLineF, rightLineF,
                       midLineFactor, yList, bottom, topLimit)
    return src, rList, yList.tolist()


def runAll(image_buffer, low_Threshold=50, height_Threshold=150, fitting_strength=8, count=20, kernel_size=3):
    src1, image_buffer1 = getSilhouette(image_buffer, low_Threshold, height_Threshold, kernel_size)
    fy1, fy2, fy3, topLimit, leftContourLimit, rightContourLimit, src2, image_buffer2 = getFinalContour(image_buffer1, fitting_strength)
    img = cv.imdecode(np.frombuffer(image_buffer2, np.uint8), cv.IMREAD_COLOR)
    src3, rList, yList = drawRadiusPic(count, img.shape[1], img.shape[0], str(fy1), str(fy2), str(fy3), topLimit, leftContourLimit, rightContourLimit)
    return src1, src2, src3, fy1, fy2, fy3, topLimit, leftContourLimit, rightContourLimit, img.shape[1], img.shape[0], rList, yList

# runAll('./images/1.jpg')
