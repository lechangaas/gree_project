"""
[INTEL CONFIDENTIAL]

Copyright (c) 2019 Intel Corporation.

This software and the related documents are Intel copyrighted materials, and
your use of them is governed by the express license under which they were 
provided to you ("License"). Unless the License provides otherwise, you may
not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express
or implied warranties, other than those that are expressly stated in the License.
"""
import cv2
import math
import numpy as np
from itertools import combinations


class MotorDetector:
    def __init__(self):
        self.dialation = 1.5
        self.dist_thresh = 5
        self.min_ratio = 1
        self.min_inner = 255
        self.min_contrast = 255
        self.min_outter = 255

        self.max_ratio = 0
        self.max_inner = 0
        self.max_contrast = -255
        self.max_outter = 0

    def distance(self, pt1, pt2):
        sqrt_dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        return sqrt_dist

    def ssim_distance(self, pair):
        # pair:(((x0, y0), (x1, y1)), ((x2, y2), (x3, y3)))
        abs_dist = abs(self.distance(pair[0][0], pair[0][1]) - self.distance(pair[1][0], pair[1][1]))
        return abs_dist

    def get_roi(self, image, cfg_param):
        # Find the coordinates of a circle with the center of two pairs of screw holes as the center and
        # half the distance between the two pairs of screw holes as the radius
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image_gray.shape
        _, image_bin = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)
        motor_circles = cv2.HoughCircles(image_bin, cv2.HOUGH_GRADIENT, dp=cfg_param.motor_dp,
                                         minDist=cfg_param.motor_minDist,
                                         param1=cfg_param.motor_param1, param2=cfg_param.motor_param2,
                                         minRadius=cfg_param.motor_minRadius, maxRadius=cfg_param.motor_maxRadius)
        circles = cv2.HoughCircles(image_bin, cv2.HOUGH_GRADIENT, dp=cfg_param.screw_hole_dp,
                                   minDist=cfg_param.screw_hole_minDist,
                                   param1=cfg_param.screw_hole_param1, param2=cfg_param.screw_hole_param2,
                                   minRadius=cfg_param.screw_hole_minRadius, maxRadius=cfg_param.screw_hole_maxRadius)

        if motor_circles is not None:
            motor_circles = list(motor_circles[0])
            motor_circles.sort(key=lambda x: self.distance((width // 2, height // 2), x))
            motor_x, motor_y, motor_inner_r = np.int0(motor_circles[0])
            motor_outter_r = int(motor_inner_r * self.dialation)

            # cv2.circle(image, (motor_x, motor_y), motor_inner_r, (255, 128, 0), 2)
            # cv2.imshow('get_roi1', image)

            # Find circles
            if circles is not None:
                circles = np.int0(circles[0])
                cand_circles = []
                for (x, y, r) in circles:
                    dist = self.distance((motor_x, motor_y), (x, y))
                    if motor_inner_r < dist < motor_outter_r:
                        cand_circles.append((x, y))
                        cv2.circle(image, (x, y), r, (128, 255, 128), -1)

                circle_pairs = []
                for pair in combinations(cand_circles, 2):
                    r0 = self.distance((motor_x, motor_y), pair[0])
                    r1 = self.distance((motor_x, motor_y), pair[1])
                    r2 = self.distance(pair[0], pair[1])
                    if abs(r0 + r1 - r2) < self.dist_thresh:
                        circle_pairs.append(pair)

                if len(circle_pairs) < 2:
                    return 3, None
                else:
                    cand_pairs = []
                    for pair in combinations(circle_pairs, 2):
                        dis_1 = self.distance(pair[0][0], pair[0][1])
                        dis_2 = self.distance(pair[1][0], pair[1][1])
                        if abs(dis_1 - dis_2) < (self.dist_thresh * 2):
                            cand_pairs.append(pair)

                    if len(cand_pairs) == 0:
                        return 2, None
                    else:
                        cand_pairs.sort(key=self.ssim_distance)
                        cand_pair = cand_pairs[0][0]
                        rr = int(
                            self.distance(cand_pair[0], cand_pair[1]) / 2)
                        xx = int((cand_pair[0][0] + cand_pair[1][0]) / 2)
                        yy = int((cand_pair[0][1] + cand_pair[1][1]) / 2)
                        cv2.circle(image, (cand_pair[0][0], cand_pair[0][1]), 8, (0, 128, 255), -1)
                        cv2.circle(image, (cand_pair[1][0], cand_pair[1][1]), 8, (0, 128, 255), -1)
                        cv2.circle(image, (xx, yy), rr, (0, 128, 255), 2)
                        cv2.circle(image, (xx, yy), motor_outter_r, (0, 128, 255), 2)
                        cv2.circle(image, (xx, yy), 8, (0, 128, 255), -1)
                        # cv2.imshow('get_roi2', image)
                        return 0, [xx, yy, rr]
            else:
                return 2, None
        else:
            return 1, None

    def get_roi_simple(self, image, cfg_param):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image_gray.shape
        _, image_bin = cv2.threshold(image_gray, 110, 100, cv2.THRESH_BINARY)
        # cv2.imshow('image_bin', image_bin)
        motor_circles = cv2.HoughCircles(image_bin, cv2.HOUGH_GRADIENT, dp=cfg_param.motor_dp,
                                         minDist=cfg_param.motor_minDist,
                                         param1=cfg_param.motor_param1, param2=cfg_param.motor_param2,
                                         minRadius=cfg_param.motor_minRadius, maxRadius=cfg_param.motor_maxRadius)

        if motor_circles is not None:
            motor_circles = list(motor_circles[0])
            motor_circles.sort(key=lambda x: self.distance((width // 2, height // 2), x))
            motor_x, motor_y, motor_inner_r = np.int0(motor_circles[0])

            # remove (near the border)
            motor_outter_r = motor_inner_r * 1.1
            x_min = motor_x - motor_outter_r
            y_min = motor_y - motor_outter_r
            x_max = motor_x + motor_outter_r
            y_max = motor_y + motor_outter_r

            # for _x, _y, _r in np.int0(motor_circles):
            #     cv2.circle(image, (_x, _y), _r, (0, 0, 255), 5)
            # print(motor_x, motor_y, motor_inner_r)
            # cv2.circle(image, (motor_x, motor_y), motor_inner_r, (255, 128, 0), 5)
            # cv2.circle(image, (motor_x, motor_y), int(motor_inner_r * 1.5), (255, 128, 0), 5)
            # cv2.imshow('get_roi', image)

            if x_min <= 0 or y_min <= 0 or x_max >= width or y_max >= height:
                # return 2, None
                return 0, [motor_x, motor_y, motor_inner_r]
            else:
                return 0, [motor_x, motor_y, motor_inner_r]
        else:
            return 1, None

    def calculate_coor(self, height, width, cx, cy, cr):
        """
        Calculate the relative coordinates of cut and pad
        Args:
            height: image height
            width:  image width
            cx: circle center x coordinate
            cy: circle center y coordinate
            cr: circle radius

        Returns:
            (cut coor), (pad coor)
        """
        cut_xmin = cx - cr if cx - cr > 0 else 0
        cut_ymin = cy - cr if cy - cr > 0 else 0
        cut_xmax = cx + cr if cx + cr < width else width
        cut_ymax = cy + cr if cy + cr < height else height

        pad_xmin = cr - (cx - cut_xmin)
        pad_ymin = cr - (cy - cut_ymin)
        pad_xmax = cr + (cut_xmax - cx)
        pad_ymax = cr + (cut_ymax - cy)
        return (cut_xmin, cut_ymin, cut_xmax, cut_ymax), (pad_xmin, pad_ymin, pad_xmax, pad_ymax)

    def crop_roi(self, image, coordinate, ratio):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image_gray.shape
        xx, yy, r_inner = coordinate
        rr = int(r_inner * ratio)
        cut_coor, pad_coor = self.calculate_coor(height, width, xx, yy, rr)

        # cut out roi
        mask = np.zeros(image_gray.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (xx, yy), rr, 1, -1)
        output_mask = image_gray * mask
        cut_xmin, cut_ymin, cut_xmax, cut_ymax = cut_coor
        valid_output = output_mask[cut_ymin:cut_ymax, cut_xmin:cut_xmax]

        # pad zero to make valid_output to be a square
        pad_output = np.zeros((2 * rr, 2 * rr), dtype=np.uint8)
        pad_xmin, pad_ymin, pad_xmax, pad_ymax = pad_coor
        pad_output[pad_ymin:pad_ymax, pad_xmin:pad_xmax] = valid_output
        return pad_output

    def find_holes(self, input_image, cfg_param):
        # vis_image = input_image.copy()

        # 找螺孔
        circles = cv2.HoughCircles(input_image, cv2.HOUGH_GRADIENT, dp=cfg_param.small_screw_hole_dp,
                                   minDist=cfg_param.small_screw_hole_minDist,
                                   param1=cfg_param.small_screw_hole_param1, param2=cfg_param.small_screw_hole_param2,
                                   minRadius=cfg_param.small_screw_hole_minRadius,
                                   maxRadius=cfg_param.small_screw_hole_maxRadius)
        if circles is not None:
            # 过滤圆
            valid_circles = []
            outter_ratio = 2
            for circle in circles[0]:
                x = int(circle[0])
                y = int(circle[1])
                r = int(circle[2])

                # 计算内圆内数据
                innermask = np.zeros(input_image.shape, dtype=np.uint8)
                innermask = cv2.circle(innermask, (x, y), r, color=255, thickness=-1)
                inner_Mean, innerStd = cv2.meanStdDev(input_image, None, None, innermask)

                # 计算内圆外侧圆环数据
                outtermask = np.zeros(input_image.shape, dtype=np.uint8)
                outtermask = cv2.circle(outtermask, (x, y), int(circle[2] * outter_ratio), color=255, thickness=-1)
                outtermask = outtermask - innermask
                outter_Mean, outter_Std = cv2.meanStdDev(input_image, None, None, outtermask)
                contrast = outter_Mean - inner_Mean

                # 设置过滤条件
                ratio = inner_Mean / outter_Mean
                flag_common = ratio < 0.62 and inner_Mean < 140 and contrast > 98 and outter_Mean > 160  # 正常螺孔螺钉
                flag_common2 = 0.55 < ratio < 0.72 and 130 < inner_Mean < 185 and outter_Mean > 241 and \
                               contrast > 75 and outter_Std < 40  # 强光螺孔
                flag_occlude = 0.56 < ratio < 0.645 and 100 < inner_Mean < 160 and 200 < outter_Mean < 240 and \
                               contrast > 80 and outter_Std > 36  # 强光下遮挡螺钉
                if flag_common or flag_common2 or flag_occlude:
                # if flag_common:
                    valid_circles.append(circle)

                    self.min_ratio = ratio if self.min_ratio > ratio else self.min_ratio
                    self.min_inner = inner_Mean if self.min_inner > inner_Mean else self.min_inner
                    self.min_contrast = contrast if self.min_contrast > contrast else self.min_contrast
                    self.min_outter = outter_Mean if self.min_outter > outter_Mean else self.min_outter

                    self.max_ratio = ratio if self.max_ratio < ratio else self.max_ratio
                    self.max_inner = inner_Mean if self.max_inner < inner_Mean else self.max_inner
                    self.max_contrast = contrast if self.max_contrast < contrast else self.max_contrast
                    self.max_outter = outter_Mean if self.max_outter < outter_Mean else self.max_outter

            #     cv2.circle(vis_image, (x, y), r, 255, 1)
            #     cv2.circle(vis_image, (x, y), int(r * outter_ratio), 255, 1)
            #     cv2.putText(vis_image, '%d %d %d %d %d' % (r, inner_Mean, contrast, outter_Mean, outter_Std),
            #                 (x, y),
            #                 cv2.FONT_HERSHEY_COMPLEX, 0.5, 180, thickness=1)
            #     print('%d %d %d %d %d' % (r, inner_Mean, contrast, outter_Mean, outter_Std))
            #     cv2.namedWindow('hough', cv2.WINDOW_NORMAL)
            #     cv2.imshow('hough', vis_image)
            # print()

            if len(valid_circles):
                return np.array(valid_circles)
            else:
                return None
        else:
            return None


if __name__ == '__main__':
    import yaml
    import sys
    import os
    from train.utils.yaml_dict import YAMLDict as ydict

    # cv2.namedWindow('image_bin', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('get_roi', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('hough', cv2.WINDOW_NORMAL)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    input_image_folder = os.path.join('/home/dls1/Desktop/20200803/error/images')
    cfg_file = os.path.join('/home/dls1/Desktop/20200803/luoding/crop-config.yaml')
    detector = MotorDetector()
    if not os.path.exists(cfg_file):
        print('config file not exist!')
        sys.exit(1)
    cfg = None
    with open(cfg_file, 'r') as stream:
        try:
            cfg = ydict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    cfg = cfg.crop_configs
    for file_name in os.listdir(input_image_folder):
        image_path = os.path.join(input_image_folder, file_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, coord = detector.get_roi_simple(image, cfg)
        if ret == 0:
            motor_x, motor_y, motor_inner_r = coord
            cv2.circle(image, (motor_x, motor_y), int(motor_inner_r), (255, 0, 0), 4)
            hole_circles = detector.find_holes(gray, cfg)
        if hole_circles is not None:
            for x, y, r in hole_circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
