import configparser
import cv2 as cv
import numpy as np
import pylab
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')

file_path = 'server\\algorithm\\config\\config.conf'


def sectionContourDraw(x, y):
    if(x == [] or y == []):
        return ''
    series = cofigfile_reader_value(
        'contourFitting', 'series')
    Factor = np.polyfit(y, x, series)
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'black', label='')
    return Factor


def pointTransform(image_width, image_height, real_width, real_height, x, y):
    real_x = real_width / image_width * x
    real_y = real_height / image_height * y
    return [real_x, real_y]


def getFinalContour(image_buffer, leftTopP, leftBottomP, rightBottomP, pic_width, pic_height, fitting_strength):
    img_org = cv.imdecode(np.frombuffer(
        image_buffer, np.uint8), cv.IMREAD_GRAYSCALE)
    image_width = img_org.shape[1]
    image_height = img_org.shape[0]
    leftTopP = pointTransform(
        pic_width, pic_height, image_width, image_height, leftTopP[0], leftTopP[1])
    leftBottomP = pointTransform(
        pic_width, pic_height, image_width, image_height, leftBottomP[0], leftBottomP[1])
    rightBottomP = pointTransform(
        pic_width, pic_height, image_width, image_height, rightBottomP[0], rightBottomP[1])
    # print(image_width, image_height, leftTopP, leftBottomP, rightBottomP)
    img_org = cv.bitwise_not(img_org)
    ret, img_bin = cv.threshold(img_org, 128, 255, cv.THRESH_TRIANGLE)

    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv.erode(img_bin, kernel, iterations=1)
    img_bin = cv.dilate(img_bin, kernel, iterations=1)

    img_thinning = cv.ximgproc.thinning(
        img_bin, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
    img_thinning = cv.ximgproc.thinning(img_org)

    image = [[], []]
    midLine = [[], []]
    leftContour = [[], []]
    rightContour = [[], []]
    img_array = np.array(img_bin-img_thinning)
    bottom = min(rightBottomP[1], leftBottomP[1])
    for i in range(img_array.shape[0]):
        if(i > img_array.shape[0] - bottom):
            break
        leftJ = 1
        rightJ = 1
        leftP = []
        rightP = []
        leftCJ = 1
        rightCJ = 1
        for x in range(img_array.shape[1]):
            y = img_array.shape[0]-i
            if(img_array[i][x] == 255):
                image[0].append(x)
                image[1].append(y)
            if(img_array[i][x] == 255 and leftCJ == 1 and y < leftTopP[1] and y > leftBottomP[1] and x > leftTopP[0]):
                leftContour[0].append(x)
                leftContour[1].append(y)
                leftCJ = 0
            if(img_array[i][img_array.shape[1] - x-1] == 255 and rightCJ == 1 and y > rightBottomP[1] and y < leftTopP[1]):
                rightContour[0].append(img_array.shape[1] - x-1)
                rightContour[1].append(y)
                rightCJ = 0
            if(img_array[i][x] == 255 and leftJ == 1 and leftCJ == 0):
                leftP = [x, y]
                leftJ = 0
            if(img_array[i][img_array.shape[1]-x-1] == 255 and rightJ == 1 and img_array.shape[1]-x-1 < rightBottomP[0] and rightCJ == 0):
                rightP = [img_array.shape[1]-x-1, y]
                rightJ = 0
            # if(leftJ == 0 and rightJ == 0):
            #     break
        if(leftP == [] or rightP == []):
            continue
        midLine[0].append((leftP[0]+rightP[0])/2)
        midLine[1].append((leftP[1]+rightP[1])/2)
    # print('leftContour:',leftContour)
    pylab.figure(figsize=(16, 9))
    pylab.plot(image[0], image[1], 'b')
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
    pylab.ylim(0, image_height)
    pylab.xlim(0, image_width)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    sio = BytesIO()
    pylab.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = str(data)
    # # 记得关闭，不然画出来的图是重复的
    pylab.close()
    sio.close()
    return fy1, fy2, fy3, message, src


def getSilhouette(image_buffer):
    img = cv.imdecode(np.frombuffer(image_buffer, np.uint8), cv.IMREAD_COLOR)
    image_width = img.shape[1]
    image_height = img.shape[0]
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new_grayImage = gray_img

    low_Threshold = cofigfile_reader_value(
        'edgeDetection', 'minthreshold')
    height_Threshold = cofigfile_reader_value(
        'edgeDetection', 'maxthreshold')
    kernel_size = cofigfile_reader_value(
        'edgeDetection', 'kemelsize')
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
    img_ndarray = np.array(imageCountour)
    newImage = np.ones(
        img_ndarray.shape, dtype=np.uint8)
    img = [[], []]
    for i in range(newImage.shape[0]):
        for j in range(newImage.shape[1]):
            if img_ndarray[i][j] < 255:
                y = newImage.shape[0] - i
                img[1].append(y)
                img[0].append(j)
                if(img_ndarray[i+1][j] > 1 and (i+1) < newImage.shape[0]/4):
                    for k in range(1, int(newImage.shape[0]/8)):
                        if(img_ndarray[i+1+k][j] < 255):
                            for l in range(k+1):
                                img_ndarray[i+1+l][j] = 0
                            break
    pylab.figure(figsize=(16, 9))
    pylab.plot(img[0], img[1], 'black')
    pylab.ylim(0, image_height)
    pylab.xlim(0, image_width)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    # pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.margins(0.0)
    # pylab.savefig(imageSavePath+"sil" +
    #               fileName+".jpg", dpi=110, bbox_inches='tight', pad_inches=0
    #               )
    sio = BytesIO()
    pylab.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = str(data)
    # # 记得关闭，不然画出来的图是重复的
    pylab.close()
    sio.close()
    return src


def dataExport(fy1, fy2, midLineFactor):
    fy1 = strToNdarray(fy1)
    fy2 = strToNdarray(fy2)
    midLineFactor = strToNdarray(midLineFactor)
    yList = np.array([600.0, 550.0, 500.0, 450.0, 400.0])
    rList = []
    for y in yList:
        dis1 = -1.0
        dis2 = -1.0
        dis = [[], []]
        normalL, x = normalLine(midLineFactor, y)
        leftLineF = fy1.copy()
        rightLineF = fy2.copy()
        p = getIntersection(leftLineF, normalL)
        dis1 = getDistance([x, y], p)
        p = getIntersection(rightLineF, normalL)
        dis2 = getDistance([x, y], p)
        if(dis1 > 0 and dis2 > 0):
            dis[0].append(dis1)
            dis[1].append(dis2)
        rList.append(dis)
    return rList


def getIntersection(factor1, factor2):
    '''
    交点计算
    输入：factor1：函数1的因数 
          factor2：函数2的因数
    输出：返回两个函数的交点坐标
    '''
    bottomLim = 250
    topLim = 800
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


def factorToPoly(Factor):
    string = ''
    for i in range(len(Factor)):
        if (str(Factor[i]))[0] != '-' and i != 0:
            string += '+'
        string += str(Factor[i])+'*y**'+str(len(Factor)-i-1)
    string += '-x'
    return string


def cofigfile_reader_value(section_name, option_name):
    # 读取配置文件
    config_reader = configparser.ConfigParser()
    config_reader.read(file_path)

    # 获取配置信息中特定的取值
    return config_reader.getint(section_name, option_name)


def configfile_revise(section_revise_list):
    # 修改配置文件
    config_writer = configparser.ConfigParser()
    config_writer.read(file_path)

    for _ in section_revise_list:

        if config_writer.has_option(_[0], _[1]):

            # 对已有的配置信息做修改
            config_writer.remove_option(_[0], _[1])
            config_writer.set(_[0], _[1], _[2])

        elif not config_writer.has_section(_[0]):

            # 添加没有的配置信息
            config_writer.add_section(_[0])
            config_writer.set(_[0], _[1], _[2])

        else:

            # 添加没有的配置信息
            config_writer.set(_[0], _[1], _[2])

    # 写入到配置文件中
    config_writer.write(open(file_path, 'w'))


# def contourToImage(imageUploadPath, fileName):
#     img_ndarray = np.array(Image.open(imageUploadPath+fileName))
#     newImage = np.ones(
#         img_ndarray.shape, dtype=np.uint8)
#     img = [[], []]
#     for i in range(newImage.shape[0]):
#         for j in range(newImage.shape[1]):
#             if img_ndarray[i][j] < 255:
#                 y = newImage.shape[0] - i
#                 img[1].append(y)
#                 img[0].append(j)
#                 if(img_ndarray[i+1][j] > 1 and (i+1) < newImage.shape[0]/4):
#                     for k in range(1, int(newImage.shape[0]/8)):
#                         if(img_ndarray[i+1+k][j] < 255):
#                             for l in range(k+1):
#                                 img_ndarray[i+1+l][j] = 0
#                             break
#     pylab.figure(figsize=(16, 9))
#     pylab.plot(img[0], img[1], 'black')
#     pylab.ylim(0, 720)
#     pylab.xlim(0, 1280)
#     pylab.xlabel('')
#     pylab.ylabel('')
#     pylab.axis('off')
#     pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
#     pylab.margins(0.0)
#     pylab.savefig(imageUploadPath+"first" +
#                   fileName[:-4]+".jpg", dpi=80
#                   # , bbox_inches='tight', pad_inches=0
#                   )
#     return np.array(img)


# def CannyThreshold(fileName, fileExtension, imageSavePath):
#     lowThreshold = 50
#     heightThreshold = 500
#     kernel_size = 3
#     imageFilePath = ""
#     imageFilePath = imageSavePath + fileName + "." + fileExtension
#     img = cv.imread(imageFilePath)
#     gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     new_grayImage = gray_img
#     detected_edges = cv.GaussianBlur(new_grayImage, (3, 3), 0)
#     detected_edges = cv.Canny(detected_edges,
#                               lowThreshold,
#                               heightThreshold,
#                               apertureSize=kernel_size)

#     Contour = cv.findContours(
#         detected_edges,
#         1,
#         1,
#     )
#     contours = Contour[0]
#     imageCountour = np.ones(detected_edges.shape, np.uint8)*255
#     cv.drawContours(imageCountour, contours, -1, (0, 255, 0), 1)
#     cv.imwrite(imageSavePath+fileName+".bmp", imageCountour)
#     drawContourImage(imageCountour, fileName)
#     # return MatrixToImage(imageCountour)


# def drawContourImage(image, fileName):
#     x = np.array([])
#     y = np.array([])

#     imageHeight = image.shape[0]
#     imageWidth = image.shape[1]
#     newImage = np.ones(
#         image.shape, dtype=np.uint8)
#     for i in range(newImage.shape[0]):
#         for j in range(newImage.shape[1]):
#             if image[i][j] < 255:
#                 y = np.append(y, imageHeight - i)
#                 x = np.append(x, j)
#     # return newImage

#     pylab.rcParams['figure.figsize'] = (16, 9)
#     plot1 = pylab.plot(x, y, 'b')
#     # plot2 = pylab.plot(x, y_pred, 'r', label='fit values')
#     pylab.title('')
#     pylab.xlabel('')
#     pylab.ylabel('')
#     pylab.axis('off')
#     # pylab.yticks([])

#     pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
#     pylab.savefig('./contourImg/'+'con'+fileName+'.jpg', dpi=80)


# def drawFirstContour(contourSavePath, fileName, leftContour, rightContour, img):
#     pylab.figure(figsize=(16, 9))
#     pylab.plot(img[0], img[1], 'b')

#     sectionContourDraw(leftContour[0], leftContour[1])
#     sectionContourDraw(rightContour[0], rightContour[1])
#     pylab.ylim(0, 720)
#     pylab.xlim(0, 1280)
#     pylab.xlabel('')
#     pylab.ylabel('')
#     pylab.axis('off')
#     pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
#     pylab.savefig(contourSavePath+fileName, dpi=80)
