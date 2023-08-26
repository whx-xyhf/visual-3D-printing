import cv2
import numpy as np


def weighted_binary_image(image):
    b, g, r = cv2.split(image)

    # 对各个通道进行二值化处理
    _, binary_b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
    _, binary_g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    _, binary_r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)

    # 加权融合得到最终的二值图像
    weight_b = 0.3  # 通道 b 的权重
    weight_g = 0.4  # 通道 g 的权重
    weight_r = 0.3  # 通道 r 的权重

    final_binary = cv2.addWeighted(binary_b, weight_b, binary_g, weight_g, 0)
    final_binary = cv2.addWeighted(final_binary, 1, binary_r, weight_r, 0)

    # 显示最终的二值图像
    cv2.imshow('Final Binary', final_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_mosaic(binary_image, k=3, valid=[]):
    if len(valid) > 0 and max(valid) >= k:
        raise Exception("max value of valid is bigger than rank")
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    contour_rank = [0]*n
    for i in range(n):
        contour = contours[i]
        contour_rank[i] = cv2.contourArea(contour)

    # 使用numpy.argsort对列表进行排序，并选择前几个最大元素的索引
    indices = np.argsort(contour_rank)[-k:]
    valid.sort(reverse=True)
    indices = [indices[i] for i in range(len(indices)) if i not in valid]
    print(indices)
    for i in indices:
        cv_contours.append(contours[i])
    # cv_contours.append(contours[6471])
    # cv2.fillPoly(binary_image, cv_contours, (255))
    # 创建空白图像用于绘制连通区域
    output_image = np.zeros_like(binary_image)

    # 绘制指定的连通区域
    # desired_contour_index = indices
    cv2.drawContours(output_image, contours,
                     indices[0], (255, 255, 255), thickness=cv2.FILLED)
    return output_image


def detect_symmetric_region(image, symmetry_lsit=[0]):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    range = int(height/200)
    result = []
    for num in symmetry_lsit:
        result.extend(range(num - range, num + range))
    print(result)
    error_list=[]
    for symmetry in result:
        if symmetry > height-symmetry:
            top_image = image[height-symmetry:symmetry, :]
            bottom_half = image[symmetry:, :]
        else:
            top_image = image[:symmetry, :]
            bottom_half = image[symmetry:2*symmetry, :]

        # 翻转左半部分并与右半部分进行比较
        flipped_top_half = cv2.flip(top_image, 0)
        error_list.append (np.abs(flipped_top_half - bottom_half))

    return symmetry_lsit[error_list.index(min(error_list))]


# def image_mosaic(binary_image, k=3, valid=[]):
#     if len(valid) > 0 and max(valid) >= k:
#         raise Exception("max value of valid is bigger than rank")
#     original_image = binary_image.copy()
#     contours, _ = cv2.findContours(
#         binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     n = len(contours)  # 轮廓的个数
#     cv_contours = []
#     contour_rank = [0]*n
#     for i in range(n):
#         contour = contours[i]
#         contour_rank[i] = cv2.contourArea(contour)

#     # 使用numpy.argsort对列表进行排序，并选择前几个最大元素的索引
#     indices = np.argsort(contour_rank)[-k:]
#     valid.sort(reverse=True)
#     indices = [indices[i] for i in range(len(indices)) if i not in valid]
#     # print(indices)
#     for i in indices:
#         cv_contours.append(contours[i])
#     # cv_contours.append(contours[6471])
#     output_image = np.zeros_like(binary_image)
#     cv2.fillPoly(output_image, cv_contours, (255))

#     return ~output_image


# 定义auto_canny函数
def auto_canny(image, sigma=0.33):
    # 计算单通道像素强度的中位数
    v = np.median(image)

    # 选择合适的lower和upper值，然后应用它们
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def show_img(image):
    # 显示结果
    cv2.imshow('Hough Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


vertical_line_list = []
horizontal_line_list = []

image = cv2.imread('resource/pic/1.jpg', 0)

# weighted_binary_image(image)

image = cv2.GaussianBlur(image, (3, 3), 0)
image = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

image = image_mosaic(image, 3, [0, 2])

edges = auto_canny(image)
vertical_lines = cv2.HoughLines(edges, 1, np.pi, threshold=120)

# 绘制检测到的直线
if vertical_lines is not None:
    for rho, theta in vertical_lines[:, 0]:
        vertical_line_list.append(int(rho))

left_offset = min(vertical_line_list)
right_row = max(vertical_line_list)
top_point_x = (left_offset+right_row)/2
horizontal_lines = cv2.HoughLines(edges[:, left_offset:right_row],
                                  1, np.pi/2, threshold=15)  # 进行霍夫直线变换

# 绘制检测到的直线
if horizontal_lines is not None:
    for rho, theta in horizontal_lines[:, 0]:
        if theta:
            horizontal_line_list.append(int(rho))
if len(horizontal_line_list) == 1:
    top_point_y = horizontal_line_list[0]
elif len(horizontal_line_list) >= 1:
    top_point_y = sum(horizontal_line_list[:2])/2
top_point = [top_point_x, top_point_y]

# 绘制检测到的直线
if horizontal_lines is not None:
    for rho, theta in horizontal_lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + left_offset + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0+left_offset - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (127), 2)  # 绘制直线

print(horizontal_line_list)
print(top_point)
show_img(image)
