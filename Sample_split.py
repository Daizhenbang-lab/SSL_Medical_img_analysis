import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.Image_Preprocessing import Mask_Extraction, Get_Skeletion, Get_Split_Information,Extract_Sample
import os
from PIL import Image
from utils.patch_extraction import zero_padding
import pandas as pd

sample = 185

Image.MAX_IMAGE_PIXELS = None
mask_path = f'full_mask_PSA/test/scan{sample}.jpg'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

height, width = mask.shape

skeleton = Get_Skeletion(mask)


skeleton_length = np.count_nonzero(skeleton)
split_length = skeleton_length / 4
split_points = []

neighbors = []

split_lines = []

split_points, split_lines = Get_Split_Information(skeleton,split_length)

image_path = f'full_sample_PSA/test/scan{sample}.jpg'
save_dir = f'mask_slides/scan{sample}'
patch_dir = f'test_set/sample/scan{sample}'
patch_path = f'test_set/sample/scan{sample}'
image = Image.open(image_path)
width, height = image.size
# sample_image = cv2.imread(image_path)
# sample_mask = Mask_Extraction(sample_image)
sample_mask = mask
#sample = Extract_Sample(sample_mask,sample_image)

# plt.imshow(sample_mask, cmap='gray')
# plt.title('Binary Image')
# plt.axis('off')
# plt.show()

# split_image1 = sample_mask.astype(np.uint8)
# split_image2 = np.zeros_like(sample_mask).astype(np.uint8)
# split_image3 = np.zeros_like(sample_mask).astype(np.uint8)
# split_image4 = np.zeros_like(sample_mask).astype(np.uint8)

# image_shape = image.shape
# sample_shape = sample_mask.shape
#
# image_col = image_shape[1]
# sample_col = sample_shape[1]
times = 1 #

#print('*************************')

def directed_area(A, B, patch_coordinate):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = patch_coordinate
    area = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
    area /= 2
    return area >= 0


# def split_images(A, B, num):
#     for y in range(split_image1.shape[0]):
#
#         for x in range(split_image1.shape[1]):
#
#             if directed_area(A, B, (x, y)) >= 0:
#                 if num == 0:
#                     split_image4[y][x] = split_image1[y][x]
#                     split_image1[y][x] = 0
#                     # split_image1[y][x][0]  = 0
#                     # split_image1[y][x][1] = 0
#                     # split_image1[y][x][2] = 0
#                 elif num == 1:
#                     split_image3[y][x]  = split_image1[y][x]
#                     split_image1[y][x] = 0
#                     # split_image1[y][x][0]  = 0
#                     # split_image1[y][x][1] = 0
#                     # split_image1[y][x][2] = 0
#                 else:
#
#                     # if y % 100 == 0 and x % 100 == 0 and :
#                     #     print((x, y))
#
#                     split_image2[y][x]  = split_image1[y][x]
#                     split_image1[y][x] = 0
#                     # split_image1[y][x][0]  = 0
#                     # split_image1[y][x][1]  = 0
#                     # split_image1[y][x][2]  = 0


patch_files = [f for f in os.listdir(patch_path) if f.endswith('.jpg') and '__x' in f and '_y' in f]

quarters = []

print(split_lines)

for num, line in enumerate(split_lines):

    files = []
    for file in patch_files:

        y_str = file.split('__x')[1].split('_')[0]
        x_str = file.split('_y')[1].split('.')[0]
        x, y = int(x_str), int(y_str)
        direct = 0
        if 'x =' in line:
            x_line = float(line.split('=')[1].strip())
            y1 = np.random.uniform(0, 10)
            y2 = np.random.uniform(y1 + 0.01, 20)
            x1 = x2 = (x_line * times)
            A = (x1, y1)
            B = (x2, y2)
            direct = directed_area(B,A,(x,y))
            #split_images(B, A, num)

        elif 'y =' in line and 'x' not in line:
            y_line = float(line.split('=')[1].strip())
            x1 = np.random.uniform(0, 10)
            x2 = np.random.uniform(x1 + 0.01, 20)
            y1 = y2 = (y_line * times)
            A = (x1, y1)
            B = (x2, y2)
            direct = directed_area(B, A, (x, y))
            #split_images(B, A, num)

        else:
            k = float(line.split('x')[0].split('=')[1].strip())
            b = float(line.split('+')[1].strip())
            y_line = lambda x: k * x + (b * times)
            x1 = np.random.uniform(0, 10)
            x2 = np.random.uniform(x1 + 0.01, 20)
            y1 = y_line(x1)
            y2 = y_line(x2)
            A = (x1, y1)
            B = (x2, y2)
            #print('----------')
            #print(line)
            #print(num)

            # if num == 2:
            #     print(A,B)
            #     print(line)

            #direct = directed_area(B, A, (x, y))

            if k > 0:
                direct = directed_area(A, B, (x, y))
            if k < 0:
                direct = directed_area(B, A, (x, y))
            #split_images(B, A, num)

        if direct:
            files.append(file)

    quarters.append(files)
    patch_files[:] = [item for item in patch_files if item not in files]

quarters.append(patch_files)
#quarters.reverse()

os.makedirs(save_dir, exist_ok=True)

# csv_dir = 'generated_result/UMAP_test_neighbor2_3D_lambda0.9.csv'
# df = pd.read_csv(csv_dir)
# #df['quarter'] = pd.NA
# for num, quarter in enumerate(quarters):
#
#     for file in quarter:
#
#         x_str = file.split('__x')[1].split('_')[0]
#         y_str = file.split('_y')[1].split('.')[0]
#         x, y = int(x_str), int(y_str)
#
#         df['x'] = df['x'].astype(int)
#         df['y'] = df['y'].astype(int)
#
#         df['sample_name'] = df['sample_name'].str.strip()
#         #print(f"Checking x: {x}, y: {y} for quarter: {num + 1}")
#         condition = (df['sample_name'] == f'scan{sample}') & (df['x'] == x) & (df['y'] == y)
#         #print(df[condition])
#         df.loc[condition, 'quarter'] = num + 1
#
# df.to_csv(csv_dir, index=False)
# print("CSV is updataed and saved。")



for num, quarter in enumerate(quarters):

    image_template = np.zeros((height, width, 3), dtype=np.uint8)
    image_template = zero_padding(image_template, 56)

    for file in quarter:

        y_str = file.split('__x')[1].split('_')[0]
        x_str = file.split('_y')[1].split('.')[0]
        x, y = int(x_str), int(y_str)


        patch = cv2.imread(os.path.join(patch_path,file))

        if patch is None:
            print(f"Error: Could not read {file}")
            continue


        patch = cv2.resize(patch, (56, 56))


        sub_patch_size = 56 // 2


        sub_patch = patch[:sub_patch_size, :sub_patch_size]


        top_left_x = int(x - sub_patch_size / 2)
        top_left_y = int(y - sub_patch_size / 2)


        if top_left_y + sub_patch_size > image_template.shape[0] or top_left_x + sub_patch_size > image_template.shape[1]:
            raise ValueError(f"the plots over boundary at ({top_left_x}, {top_left_y}) ，"
                             f"sub_patch_size: {sub_patch_size}, "
                             f"image_template size: {(height, width)}")
        else:
            image_template[top_left_y:top_left_y + sub_patch_size, top_left_x:top_left_x + sub_patch_size] = sub_patch

    cv2.imwrite(f'{save_dir}/image_template_quarter_{num}.jpg', image_template)
    print(f'Quarter {num} processed and saved as image_template_quarter_{num}.jpg')




# save_dir = 'mask_slides/scan18'
# os.makedirs(save_dir, exist_ok=True)
# base_name = os.path.basename(image_path).split('.')[0]
# file_template = os.path.join(save_dir, f'{base_name}_{{}}.jpg')
#
# # cv2.imwrite(file_template.format(1), split_image1)
# # cv2.imwrite(file_template.format(2), split_image2)
# # cv2.imwrite(file_template.format(3), split_image3)
# # cv2.imwrite(file_template.format(4), split_image4)
#
# plt.imsave(file_template.format(1), split_image1, cmap='gray')
# plt.imsave(file_template.format(2), split_image2, cmap='gray')
# plt.imsave(file_template.format(3), split_image3, cmap='gray')
# plt.imsave(file_template.format(4), split_image4, cmap='gray')
#
#
# print("Images have been saved successfully.")

