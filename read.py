import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import matplotlib.pyplot as plt

I = Image.open('D:/PyCharm Project/CNN/2.png')
# I_array = np.array(I)
# index = np.argwhere(I_array ==2)
# y = index[0:-1,0:1]
# uniques = np.unique(y)
# a=np.arange(28687)
# y_1 = uniques-a
# a_1 = np.unique(y_1)
# index_1 = np.argwhere(y_1 == 21014)



box = (0,3694,2337,3959)
roi = I.crop(box)
roi.save('d:/02.png')