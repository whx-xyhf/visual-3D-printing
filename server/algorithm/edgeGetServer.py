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
from scipy.integrate import quad
matplotlib.use('Agg')


def factorDiff(Factor):
    """
    因数导数计算
    """
    returnArray = []
    length = len(Factor)-1
    for i in range(length):
        returnArray.append(Factor[i]*(length-i))
    returnArray = np.array(returnArray)
    return returnArray


def getContours(inputImageArray):
    leftContour = [[], []]
    rightContour = [[], []]
    j = 0
    for i in range(len(inputImageArray[0])-1):
        if inputImageArray[0][i] != inputImageArray[0][i+1]:
            leftContour[0].append(inputImageArray[1][j])
            leftContour[1].append(inputImageArray[0][j])
            rightContour[0].append(inputImageArray[1][i])
            rightContour[1].append(inputImageArray[0][i])
            j = i+1
    leftContour[0].append(inputImageArray[1][j])
    leftContour[1].append(inputImageArray[0][j])
    rightContour[0].append(inputImageArray[1][-1])
    rightContour[1].append(inputImageArray[0][-1])
    return leftContour, rightContour


def sectionContourDraw(x, y, fitting_strength):
    Factor = np.polyfit(y, x, fitting_strength)
    drawFunction(Factor, y)
    return Factor


def drawFunction(Factor, y):
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'black', label='')


def getTopLimit(leftContour, rightContour, compareNum=4):
    '''
    上边界判定
    输入图像矩阵
    输出上边界y值
        '''
    def relativeSlope(inputList):
        '''
        # 相对斜率
        输入为列表
        返回为数组
        '''
        relativeSlopeList = []
        for i in range(len(inputList)-1):
            relativeSlopeList.append(
                (inputList[i]-inputList[i+1])/(inputList[i+1]+1))
        return np.array(relativeSlopeList)

    pixelStatistics = []
    minLen = min(len(leftContour[0]), len(rightContour[0]))
    for i in range(minLen):
        pixelStatistics.append(leftContour[0][i]-rightContour[0][i])

    der1Contour = relativeSlope(pixelStatistics)

    topLimit = 0
    for i in np.array(signal.argrelextrema(der1Contour, np.greater, order=10)[0])[:compareNum]:
        if der1Contour[i] > der1Contour[topLimit]:
            topLimit = i

    return topLimit+1


def splitArray(inputYArray, inputXArray):
    returnArray = [[], []]
    n = 0
    for i in range(len(inputYArray)-1):
        if inputYArray[i] != inputYArray[i+1]:
            returnArray[0].append(inputXArray[n:i+1])
            returnArray[1].append(inputYArray[n:i+1])
            n = i+1
    returnArray.append(inputYArray[n:])
    return returnArray


def imageArrayToStr(inputImgArray):
    tempImg = cv.imencode(
        '.png', inputImgArray)[1]
    returnImgStr = str(base64.encodebytes(tempImg).decode())
    return returnImgStr


def strToGrayImgArray(inputImgStr):
    return cv.imdecode(np.frombuffer(inputImgStr, np.uint8), cv.IMREAD_GRAYSCALE)


# Get image silhouette
def getSilhouette(image_buffer, low_Threshold=50, height_Threshold=150, kernel_size=5):

    def auto_canny(image, lowersigma=0.33, uppersigma=0.2):
        # 计算单通道像素强度的中位数
        v = np.median(image)
        # 选择合适的lower和upper值，然后应用它们
        lower = int(max(0, (1.0 - lowersigma) * v))
        upper = int(min(255, (1.0 + uppersigma) * v))
        edged = cv.Canny(image, lower, upper)
        return edged

    def getBottomLineByColumn(pixelStatistics):
        """
        pixelStatistics:input array
        """
        pixelStatistics = savgol_filter(pixelStatistics, 5, 3)
        der1Contour = np.diff(pixelStatistics)
        leftFirstMutation = np.array(
            signal.argrelextrema(der1Contour, np.greater)[0])[0]
        leftContourLimit = list(
            Counter(leftContour[0][:leftFirstMutation]).keys())[0]
        return leftContourLimit

    def getBottomLineByContour(leftContourXArray):
        return leftContourXArray.index(max(leftContourXArray))

    def imgPatch(img, leftC, rightC, compareNum):
        """
        上部缺口修补
        """
        offset = int(len(img[1])/80)

        topLimit = getTopLimit(leftContour, rightContour, compareNum)
        while topLimit+2*offset > len(leftC[1]) and compareNum >= 0:
            compareNum -= 1
            topLimit = getTopLimit(leftContour, rightContour, compareNum)
        if not compareNum:
            return 0

        # begin = topLimit-offset
        # if topLimit < offset or begin < offset:
        #     begin = offset

        # for row in range(begin, topLimit+offset):
            left = max(min(leftC[0][row-offset:row]),
                       min(leftC[0][row+1:row+offset]))
            right = min(max(rightC[0][row-offset:row]),
                        max(rightC[0][row+1:row+offset]))
            if leftC[0][row] > left:
                img[0][row*2] = left
            if rightC[0][row] < right:
                img[0][row*2+1] = right
        return topLimit

    def contourFill(img, xArry, yArry, limit=400):
        contourImgFill = np.ones(img.shape, np.uint8)*255
        for i in range(0, len(xArry)-1, 2):
            contourImgFill[yArry[i]][xArry[i]:xArry[i+1]] = 0
        # print(len(xArry)-1,i)
        return contourImgFill

    def pointConnect(inputXArray, inputYArray, limit):
        returnArray = [inputXArray[:limit], inputYArray[:limit]]
        precentDis = 0
        precentIndex = limit+1
        minIndex = 0
        writeIndex = limit
        minDis = np.inf
        arrayLength = len(inputXArray)
        while precentIndex < arrayLength:
            precentDis = (inputXArray[precentIndex]-inputXArray[writeIndex])**2+(
                inputYArray[precentIndex]-inputYArray[writeIndex])**2
            if minDis > precentDis:
                minDis = precentDis
                minIndex = precentIndex
            if (precentIndex-writeIndex+1)**2 > minDis or precentIndex+1 == arrayLength:
                step = (inputXArray[minIndex]-returnArray[0]
                        [-1])/(minIndex-writeIndex+1)
                baseValue = returnArray[0][-1]
                for i in range(1, minIndex-writeIndex+1):
                    if int(baseValue+i*step) < inputXArray[writeIndex+i]:
                        returnArray[0].append(inputXArray[writeIndex+i])
                        returnArray[1].append(inputYArray[writeIndex+i])
                        writeIndex = writeIndex+i
                        minDis = np.inf
                        minIndex = writeIndex+1
                        precentIndex = writeIndex
                        reset = True
                        break
                    returnArray[0].append(int(baseValue+i*step))
                    returnArray[1].append(inputYArray[writeIndex+i])
                if "reset" in locals() and reset:
                    reset = False
                    continue
                else:
                    writeIndex = minIndex
                    minDis = np.inf
                    minIndex += 1
                    precentIndex = writeIndex
            precentIndex += 1
        return returnArray

    def refineContour(img, xArry, yArry):
        contourImgFill = np.ones(img.shape, np.uint8)*255
        contourImgFill = ~contourImgFill
        refineContourImg = contourImgFill.copy()
        returnArray = [[], []]
        targetNum = 255
        for i in range(0, len(yArry), 2):
            if targetNum in img[yArry[i]][xArry[i]:xArry[i+1]]:
                returnArray[0].append(list(img[yArry[i]]
                                           [xArry[i]:xArry[i+1]]).index(targetNum)+xArry[i]+1)
                returnArray[1].append(yArry[i])
            if targetNum in img[yArry[i]][xArry[i+1]:xArry[i]:-1]:
                returnArray[0].append(
                    xArry[i+1]-list(img[yArry[i]][xArry[i+1]:xArry[i]:-1]).index(targetNum)-1)
                returnArray[1].append(yArry[i])
            if len(returnArray[0]) == 0:
                break
            contourImgFill[yArry[i]][xArry[i]:xArry[i+1]] = 255
            refineContourImg[yArry[i]][returnArray[0]
                                       [-2]:returnArray[0][-1]] = 255

        # returnImg = contourImgFill-refineContourImg
        returnImg = ~refineContourImg
        return returnArray, returnImg

    def showMaskImg(inputImg, inputXArray, inputYArray):
        img = inputImg.copy()
        fillColor = 0
        for i in range(0, len(inputYArray), 2):
            img[inputYArray[i]][:inputXArray[i]] = fillColor
            img[inputYArray[i]][inputXArray[i+1]+1:] = fillColor
        return img

    orgImg = cv.imdecode(np.frombuffer(
        image_buffer, np.uint8), cv.IMREAD_GRAYSCALE)
    image_width = orgImg.shape[1]
    image_height = orgImg.shape[0]
    detected_edges = cv.GaussianBlur(orgImg, (3, 3), 0)
    detected_edges = cv.Canny(detected_edges,
                              low_Threshold,
                              height_Threshold,
                              apertureSize=kernel_size)
    contours = cv.findContours(detected_edges, 1, 1,)[0]
    imageCountour = np.ones(detected_edges.shape, np.uint8)*255
    cv.drawContours(imageCountour, contours, -1, (0), cv.FILLED)
    imageCountour = cv.threshold(
        imageCountour, 127, 255, cv.THRESH_BINARY)[1]

    grayImage = np.array(imageCountour)
    xy = list(np.where(grayImage < 1))
    leftContour, rightContour = getContours(xy)
    pixelStatistics = abs(np.asarray(
        rightContour[0])-np.asarray(leftContour[0]))
    der1Contour = np.diff(pixelStatistics)
    for i in signal.argrelextrema(der1Contour, np.greater)[0]:
        if der1Contour[i] > 20:
            rightContourLimit = i+1
            break
    if "rightContourLimit" not in locals():
        rightContourLimit = rightContour[1][-1]

    towardsFlag = False
    # right = rightContour[0][rightContourLimit]
    if leftContour[0].index(max(leftContour[0])) < rightContourLimit:
        grayImage = np.array(imageCountour[:, ::-1])
        xy = list(np.where(grayImage < 1))
        leftContour, rightContour = getContours(xy)
        towardsFlag = True

    rightContour = pointConnect(
        rightContour[0], rightContour[1], rightContourLimit+1)
    xy = [[], []]
    xy[0] = list(itertools.chain.from_iterable(
        zip(leftContour[0], rightContour[0])))
    xy[1] = list(itertools.chain.from_iterable(
        zip(leftContour[1], rightContour[1])))
    # xy, refineContourImg = refineContour(imageCountour, xy[0], xy[1])
    compareNum = 4
    topLimit = imgPatch(xy, leftContour, rightContour, compareNum)
    # left = xy[0][topLimit]-30

    leftContourLimit1 = getBottomLineByContour(leftContour[0])
    leftContourLimit2 = getBottomLineByColumn(pixelStatistics)

    if leftContourLimit1 > rightContourLimit and leftContourLimit2 > rightContourLimit:
        leftContourLimit = min(leftContourLimit1, leftContourLimit2)
    elif leftContourLimit1 > rightContourLimit and leftContourLimit2 < rightContourLimit:
        leftContourLimit = leftContourLimit1
    else:
        leftContourLimit = leftContourLimit2
    leftContour[0] = leftContour[0][:leftContourLimit]
    leftContour[1] = leftContour[1][:leftContourLimit]

    if towardsFlag:
        orgImg = orgImg[:, ::-1]
    tempImg = contourFill(orgImg, xy[0], xy[1])

    # # # contourImg=contourFill(imageCountour,xy[0],xy[1])
    # cv.imwrite("./newimg.png", tempImg[:leftContourLimit, 700:1300])
    # cv.imshow("", cv.resize(tempImg[:leftContourLimit, :],
    #           [image_width//2, image_height//2]))
    # cv.waitKey(0)

    for i in range(len(xy[1])):
        if xy[1][i] >= leftContourLimit:
            xy[0] = xy[0][:i]
            xy[1] = xy[1][:i]
            break
    pylab.axis('equal')
    pylab.plot(xy[0],  xy[1], 'black')
    pylab.ylim(image_height, 0)
    pylab.xlim(0, image_width)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    pylab.margins(0.0)
    sio = BytesIO()
    pylab.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    pylab.close()
    silhouetteImg = sio.getvalue()
    sio.close()

    return imageCountour, tempImg[:leftContourLimit, :], topLimit, towardsFlag


# Draw Fitting Line
def getFinalContour(image_buffer,  fitting_strength, topLimit, count):

    def rankMax(rank, nums):
        """
        :type nums: List[int]
        :rtype: int
        :rank
        """
        nums = list(set(nums))
        nums.sort()
        if len(nums) < rank:
            return nums[-1]
        else:
            return nums[-rank]

    def normalLine(factor, y):
        f = np.poly1d(factor)
        x = f(y)
        derF = f.deriv(1)
        derX = derF(y)
        ky = -1/derX
        const = x-ky*y
        lineFactor = np.array([ky, const])
        return lineFactor, x, derX

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

    def getDistance(p1, p2):
        return (((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5).item()

    def getIntersection(factor1, factor2, bottomLim, topLim):
        '''
        交点计算
        输入：\n
            factor1：函数1的因数\n
            factor2：函数2的因数\n
            bottomLim：下边界限制\n
            topLim：上边界限制\n
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

    def drawNormalLine(yFactor, x):
        xFactor = np.array([1/yFactor[0], -yFactor[1]/yFactor[0]])
        F = np.poly1d(xFactor)
        fY = F(x)
        pylab.plot(x, fY,  'red', label='')

    def arcLengthCompute(factor, lowerBound, heightBound):

        def factorToPoly(Factor):
            '''
            因数转为表达式字符串
            输入：Factor：函数的因数
            输出：函数表达式字符串
            '''
            string = ''
            for i in range(len(Factor)):
                if (str(Factor[i]))[0] != '-' and i != 0:
                    string += '+'
                string += str(Factor[i])+'*y**'+str(len(Factor)-i-1)
            return string

        """
        弧长计算
        输入：
            factor：函数因数 
            lowerBound：积分下界
            heightBound：积分上界
        输出：
            弧长
            误差范围
        """

        der1f1 = factorDiff(factor)
        f = factorToPoly(der1f1)
        return quad(lambda y: (1+eval(f)**2) ** 0.5,
                    lowerBound, heightBound)

    def ObliqueLength(slope):
        if slope > 1:
            return (1+(1/slope)**2)**0.5
        return (1+slope**2)**0.5

    def getNozzleDiameter(inputArray):
        inputArray.sort()
        mid = len(inputArray)//2
        return (inputArray[mid]+inputArray[~mid])//2
    # img_org = strToGrayImgArray(image_buffer)
    img_org = image_buffer
    img_org = ~img_org

    # 原图像已取反，故原区域现为白色
    img = np.array(img_org)
    xy = list(np.where(img > 253))
    leftContour, rightContour = getContours(xy)
    topLimit = min(getTopLimit(leftContour, rightContour), topLimit)
    pixelStatistics = np.asarray(rightContour[0])-np.asarray(leftContour[0])
    leftContourLimit = leftContour[1][-1]
    bottom = leftContourLimit+30
    pixelDiff = np.diff(pixelStatistics)
    rightContourDiff = np.diff(rightContour[0])

    # # # get rightContourLimit
    # it must be the first greater point or the least point
    for i in signal.argrelextrema(pixelDiff, np.greater)[0]:
        if pixelDiff[i] > 20 and i in signal.argrelextrema(rightContourDiff, np.greater)[0] and i > topLimit:
            rightContourLimit = i
            break
    if "rightContourLimit" not in locals():
        rightContourLimit = rightContour[1][-1]

    for i in range(len(rightContour[0])):
        if rightContour[1][i] >= topLimit:
            rightContour[0] = rightContour[0][topLimit:rightContourLimit]
            rightContour[1] = rightContour[1][topLimit:rightContourLimit]
            break
    for i in range(len(leftContour[0])):
        if leftContour[1][i] >= topLimit:
            leftContour[0] = leftContour[0][topLimit:]
            leftContour[1] = leftContour[1][topLimit:]
            break
    leftContour[0].append(rightContour[0][-1])
    leftContour[1].append(leftContour[1][-1])

    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv.threshold(img_org, 128, 255, cv.THRESH_TRIANGLE)[1]
    img_bin = cv.erode(img_bin, kernel, 1)
    img_bin = cv.dilate(img_bin, kernel, 1)
    img_thinning = cv.ximgproc.thinning(
        img_bin, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
    img_thinning = cv.ximgproc.thinning(img_org)
    skeletonImg = ~(img_org-img_thinning)
    midLine = [[], []]
    # 返回值中0为纵坐标，1为横坐标
    midLineXY = np.where(img_thinning == 255)

    midLine[0].append(midLineXY[1][0])
    midLine[1].append(midLineXY[0][0])
    for i in range(1, len(midLineXY[1])-1):
        if midLineXY[0][i] != midLine[1][-1]:
            midLine[0].append(midLineXY[1][i])
            midLine[1].append(midLineXY[0][i])

    for i in range(len(midLine[0])):
        if midLine[0][i] > rightContour[0][-1] and midLine[1][i] > rightContour[1][-1]:
            midLine[0] = midLine[0][:i]
            midLine[1] = midLine[1][:i]
            break

    # midLIne Upper end interception
    for i in range(len(midLine[1])):
        if midLine[1][i] >= topLimit:
            midLine[0] = midLine[0][i:]
            midLine[1] = midLine[1][i:]
            break

    midLineLimit = midLine[1][-1]

    left = min(leftContour[0]) - 30
    if left < 0:
        left = 0
    right = max(rightContour[0]) + 30
    if right > len(img[0]):
        right = len(img[0])

    leftContourFactor = sectionContourDraw(
        leftContour[0], leftContour[1], fitting_strength)
    rightContourFactor = sectionContourDraw(
        rightContour[0], rightContour[1], fitting_strength)
    midContourFactor = sectionContourDraw(
        midLine[0], midLine[1], fitting_strength)

    FL = np.poly1d(leftContourFactor)
    fXL = FL(topLimit)
    FR = np.poly1d(rightContourFactor)
    fXR = FR(topLimit)
    pylab.plot([fXL, fXR], [topLimit, topLimit],  'blue', label='')
    yList = np.linspace(topLimit, midLineLimit, count, endpoint=False)[1:]
    nozzleDiameter = getNozzleDiameter(pixelStatistics[:topLimit])
    dList = []
    dList.insert(0, fXR - fXL)
    arcLength = [0]
    for y in yList:
        normalLineFactor, x, slope = normalLine(midContourFactor, y)
        leftIntersection = getIntersection(
            leftContourFactor.copy(), normalLineFactor, leftContourLimit, topLimit)
        rightIntersection = getIntersection(
            rightContourFactor.copy(), normalLineFactor, leftContourLimit, topLimit)
        if leftIntersection[1] == 0 or rightIntersection[1] == 0:
            continue
        dis1 = getDistance([x, y], leftIntersection)
        dis2 = getDistance([x, y], rightIntersection)
        dList.append(dis1+dis2)
        if leftIntersection[0] < rightIntersection[0]:
            xRange = (leftIntersection[0], rightIntersection[0])
        else:
            xRange = (rightIntersection[0], leftIntersection[0])
        drawNormalLine(normalLineFactor, xRange)
        arcLength.append(arcLengthCompute(midContourFactor, topLimit, y)[0])

    pylab.axis('equal')
    pylab.ylim(bottom, 0)
    pylab.xlim(left, right)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    sio = BytesIO()
    pylab.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    pylab.close()
    value = sio.getvalue()
    sio.close()
    img = cv.imdecode(np.frombuffer(value, np.uint8), cv.IMREAD_COLOR)
    scale = img.shape[0]/bottom
    width = right-left
    finalImg = img[:, int(img.shape[1]/2-scale*width/2)
                          :int(img.shape[1]/2+scale*width/2)]
    skeletonImg = skeletonImg[:bottom, left:right]
    return skeletonImg, finalImg, nozzleDiameter, dList, arcLength


def runAll(sourceImageBuffer, low_Threshold=50, height_Threshold=150, fitting_strength=8, count=20, kernel_size=3):
    imageCountour, silhouetteImg, getToplimit, towardsFlag = getSilhouette(
        sourceImageBuffer, low_Threshold, height_Threshold, kernel_size)
    skeletonImg, finalImg, nozzleDiameter, dList, arcLength = getFinalContour(
        silhouetteImg, fitting_strength, getToplimit, count)
    if towardsFlag:
        skeletonImg = skeletonImg[:, ::-1]
        finalImg = finalImg[:, ::-1]
    cv.imwrite("skeletonImg.png", skeletonImg)
    cv.imwrite("imageCountour.png", imageCountour)
    cv.imwrite("finalImg.png", finalImg)

    return imageArrayToStr(imageCountour), imageArrayToStr(skeletonImg), imageArrayToStr(finalImg), dList, arcLength, nozzleDiameter
