import cv2
import numpy as np

from skimage.morphology import skeletonize
import skimage.morphology
from skimage.util import img_as_ubyte
from fil_finder import FilFinder2D
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
def Mask_Extraction (image):



    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_thres = np.array([50, 10, 100])
    upper_thres = np.array([180, 255, 255])

    mask = cv2.inRange(hsv_image, lower_thres, upper_thres)
    blue_result = cv2.bitwise_and(image, image, mask=mask)

    gray_image = cv2.cvtColor(blue_result, cv2.COLOR_HSV2BGR)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, thresh=1, maxval=1, type=cv2.THRESH_BINARY)

    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_img = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)


    contours, _ = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)




    area_limit = 1000

    cleaned_image = np.zeros_like(opened_img)

    for contour in contours:
        if cv2.contourArea(contour) > area_limit:
            cv2.drawContours(cleaned_image, [contour], -1, (255), thickness=cv2.FILLED)

    cleaned_image = cleaned_image > 0

    return cleaned_image


def Get_Skeletion(mask):

    skeleton = skimage.morphology.skeletonize(mask)
    skeleton = img_as_ubyte(skeleton)

    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

    return fil.skeleton_longpath


def Get_Split_Information(skeleton, split_length):

    split_points = []
    neighbors = []
    split_lines = []
    count = 0

    for row_index, row in enumerate(skeleton):

        if np.any(row != 0):

            for col_index in range(len(row) - 1, -1, -1):
                if row[col_index] != 0:
                    count += 1
                    neighbors.insert(0, [row_index, col_index])

                    if len(neighbors) >= 9:
                        neighbors.pop()

                        if split_points and neighbors[3] == split_points[-1]:
                            x = [neighbor[1] for neighbor in neighbors]
                            y = [neighbor[0] for neighbor in neighbors]

                            if np.all(x == x[0]):
                                line_description = f"y = {neighbors[4][0]}"
                            elif np.all(y == y[0]):
                                line_description = f"x = {neighbors[4][1]}"

                            else:
                                coefficients = np.polyfit(x, y, 8)

                                poly_der = np.polyder(coefficients)

                                k = np.polyval(poly_der, neighbors[4][1])
                                k = -1.0 / k
                                b = neighbors[4][0] - k * neighbors[4][1]
                                line_description = f"y = {k:.2f}x + {b:.2f}"

                            split_lines.append(line_description)

                    if count >= split_length:
                        split_points.append([row_index, col_index])
                        count = 0

    return split_points, split_lines

def Extract_Sample (mask, image):

    extracted_sample = np.zeros_like(image)
    extracted_sample[mask > 0] = image [mask > 0]

    return extracted_sample

