import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

I = Image.open('./image_1.png')
I_array = np.array(I)
print(I_array)
img = cv2.imread('D:/PyCharm Project/CNN/image_1.png')
# cv2.imshow('image',img)
# cv2.waitKey(0)
# px = img[100,100]
# print(px)