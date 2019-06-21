import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

I = Image.open('./image_1.png')
I_array = np.array(I)
print(I_array)
index = np.argwhere(I_array ==1)
y = index[0:-1,0:1]
uniques = np.unique(y)
a=np.arange(28687)
y_1 = uniques-a
a_1 = np.unique(y_1)
index_1 = np.argwhere(y_1 == 11062)
print(index.shape)
# img = cv2.imread('D:/PyCharm Project/CNN/image_1.png')
# cv2.imshow('image',img)
# cv2.waitKey(0)
# px = img[100,100]
# print(px)