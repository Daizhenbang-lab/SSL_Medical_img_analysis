import cv2


get_max = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        h, s, v = hsv[y, x]

        get_max.append([h,s,v])
        print(f'H:{h}, S:{s}, V:{v}')


img = cv2.imread(r'images/scan17.ndpi - Series 3.jpg')
cropImg = img[2944:3145, 1694:1981]
resized_image = cv2.resize(cropImg, (1500, 1000))
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

img_name = 'image'
cv2.namedWindow(img_name)
cv2.setMouseCallback(img_name, mouse_callback)

cv2.imshow(img_name, resized_image)
cv2.waitKey(0)

h_list = [value[0] for value in get_max]
s_list = [value[1] for value in get_max]
v_list = [value[2] for value in get_max]

print('h maximum：',max(h_list))
print('s maximum：',max(s_list))
print('v maximum：',max(v_list))

print('h minimum：',min(h_list))
print('s minimum：',min(s_list))
print('v minimum：',min(v_list))

# row = [point[0]  for point in split _points]
# col = [point[1]  for point in split_points]