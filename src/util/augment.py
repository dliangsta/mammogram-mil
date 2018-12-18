# adapted from https://gist.github.com/tomahim/9ef72befd43f5c106e592425453cb6ae 
import os
import random
from shutil import copyfile
import numpy as np

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

INPUT_DATA_PATH = '../../data/ddsm'
OUTPUT_DATA_PATH = '../../data/augmented_data'
ROTATION_DEGREES = [0, 90, 180, 270]
MAX_CROP_SIZE = 50
SEED = 42
IMG_EXT = '.png'

def split_name(filename):
  name = os.path.splitext(filename)[0] 
  ext = os.path.splitext(filename)[1]
  return name, ext

def rotate(img, degree):
  new_img = sk.transform.rotate(img, degree)
  return new_img

def zoom(img):
  width, height = img.shape
  crop_size = random.randint(0, MAX_CROP_SIZE)
  crop_width = width - crop_size
  crop_height = height - crop_size
  new_img = img[0:crop_width, 0:crop_height]
  new_img = sk.transform.resize(new_img, (width, height))
  return new_img

# loop over all the types of cases, then loop over all the cases
# for each case create 15 new cases: 
# - 3 rotations (90, 180, and 270 degree) per each of the 4 images
# - 1 rescale (zoom in) per each of the 4 images
random.seed(SEED)
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

        if case_file.endswith(IMG_EXT):
          img_name, img_ext = split_name(case_file)
          # augment png
          case_file_to_transform = sk.io.imread(case_file_path)

          # no flip, up-down, left-right, up-down and left-right
          flips = [('none', lambda x: x), ('ud', np.flipud), ('lr', np.fliplr), ('udlr', lambda x: np.fliplr(np.flipud(x)))]

          for rotation in ROTATION_DEGREES:
            for flip, flip_func in flips:
              for zoom_idx in range(3):
                new_name = "{}_{}{}_{}{}_{}{}{}".format(img_name, "rotated", rotation, "flipped", flip, "zoomed", zoom_idx, img_ext)
                new_path = os.path.join(OUTPUT_DATA_PATH, category, case_dir, new_name)
                if not os.path.exists(new_path):
                  image = rotate(case_file_to_transform, rotation)
                  image = flip_func(image)
                  if zoom_idx > 0:
                    image = zoom(image) 
                  print(new_path)
                  sk.io.imsave(new_path, image) 
        else:
          # just copy it over with same name
          copyfile(case_file_path, new_case_file)
