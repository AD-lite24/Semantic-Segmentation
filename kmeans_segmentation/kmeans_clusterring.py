import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
# from sklearn.cluster import Kmeans

orig_img = cv2.imread('/Users/adityadandwate/Desktop/Projects/Proper/sem_seg/images/house.jpg')
orig_img = cv2.resize(orig_img, (640, 480))
img=cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 6
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))
result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

# clt = Kmeans(n_clusters = 5)
# clt.fit(img)

figure_size = 8
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()