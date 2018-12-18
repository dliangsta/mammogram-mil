# adapted from https://gist.github.com/tomahim/9ef72befd43f5c106e592425453cb6ae 
import os
import random
from shutil import copyfile
import numpy as np
try:
    import opencv2 as cv2
except:
    import cv2

INPUT_DATA_PATH = '../../data/ddsm'
OUTPUT_DATA_PATH = '../../data/ddsm_cropped'
for type in os.listdir(INPUT_DATA_PATH):
  for category in os.listdir(INPUT_DATA_PATH): 
    category_dir = os.path.join(INPUT_DATA_PATH, category)
    case_dirs = os.listdir(category_dir)

    for case_dir in case_dirs:
      for case_file in os.listdir(os.path.join(category_dir, case_dir)):
        case_file_path = os.path.join(INPUT_DATA_PATH, category, case_dir, case_file) 
        new_case_path = os.path.join(OUTPUT_DATA_PATH, category, case_dir)
        new_case_file = os.path.join(new_case_path, case_file)
        if (not os.path.exists(new_case_path)):
          os.makedirs(new_case_path)

        if case_file.endswith('.png'):
            img = cv2.imread(case_file_path)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            out = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            print(len(out))
            for i in out:
                print(i)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            crop = img[y:y+h,x:x+w]
            cv2.imshow(crop)
            1/0
            # cv2.imwrite('sofwinres.png',crop)