import glob

import cv2

root_path = '/home/coolshen/Desktop/code/mycode/LSTM_GAN/my_data/download/W. North Pacific'

img_list = sorted(glob.glob(f'{root_path}/*/*/*jpg'))

for _,i in enumerate(img_list):
    img = cv2.imread(i)
    cv2.imshow('my_img',img)
    cv2.waitKey(0)