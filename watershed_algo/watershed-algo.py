import numpy as np
import cv2
import skimage.io as io 
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread('/Users/adityadandwate/Desktop/Projects/Proper/sem_seg/images/test_img.jpg')
print(type(img))
cv2.imshow('orignal', img)


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
plt.figure()
cv2.imshow('', sure_bg)
plt.imsave('test.jpg', sure_bg)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)
plt.figure()
plt.imshow(sure_fg)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
plt.figure()
cv2.imshow('border', unknown)


orig_img = cv2.imread('/Users/adityadandwate/Desktop/Projects/Proper/sem_seg/images/test_img.jpg')
mask = cv2.imread('test.jpg')

mask[mask < 128] = 0
mask[mask > 128] = 1
output_img = orig_img*(mask == 1)
output_img[mask == 0] = 255
cv2.imshow('output', output_img)


cv2.waitKey(0)
cv2.destroyAllWindows()


