import os
import random
import math
import shutil
import cv2
import numpy as np
import csv
random.seed(42)


def main():
  train_split, dev_split, test_split = 0.9, 0.05, 0.05
  input_dirs = ["../../data/ddsm", "../../data/ddsm_large"]
  output_dirs = ["../../data/{}_case_images", "../../data/{}_case_images_large"]
  label_map = {"normals": 0, "benigns": 0, "benign_without_callbacks": 0, "cancers": 1}
  # label_map = {"benigns": 0, "benign_without_callbacks": 0, "cancers": 1}

  assert (train_split + dev_split + test_split) == 1

  for operation in [stack, stitch]:
    for input_dir, output_dir in zip(input_dirs, output_dirs):
      if "large" in input_dir:
        continue
      output_dir = output_dir.format("stack" if stack == operation else "stitch")
      if not os.path.exists(os.path.join(output_dir, "images")):
        os.makedirs(os.path.join(output_dir, "images"))
      train_paths_labels = []
      dev_paths_labels = []
      test_paths_labels = []
      # Loop through each classification category. Preserve the split ratio for each category.
      for category in os.listdir(input_dir):
        if category not in label_map:
          continue
        print(category)
        category_dir = os.path.join(input_dir, category)
        case_dirs = os.listdir(category_dir)

        # Make sure each case stays in the same set.
        case_dirs = [case_dir for case_dir in case_dirs]
        random.shuffle(case_dirs)

        # Creates list of tuples (path, label)
        output_paths_labels = []
        for case_dir in case_dirs:
          case_images = []
          for case_file in os.listdir(os.path.join(category_dir, case_dir)):
            if case_file.endswith(".png"):
              case_images.append(os.path.join(category_dir, case_dir, case_file))
          case_images.sort()
          if len(case_images) == 4:
            if random.random() < .6 or "cancers" in category:
              output_path = os.path.join(output_dir, "images", case_dir + ".png")
              if not os.path.exists(output_path):
                assert 'LEFT_CC' in case_images[0]
                assert 'LEFT_MLO' in case_images[1]
                assert 'RIGHT_CC' in case_images[2]
                assert 'RIGHT_MLO' in case_images[3]
                image = operation(case_images)
                cv2.imwrite(output_path, image)
              output_paths_labels.append((output_path.replace('../',''), label_map[category]))
          else:
            print("Skipping %s because it has %d images." % (os.path.join(category_dir, case_dir), len(case_images)))

        train_paths_labels.extend(output_paths_labels[:int(math.floor(len(output_paths_labels) * train_split))])
        dev_paths_labels.extend(output_paths_labels[int(math.floor(len(output_paths_labels) * train_split)) : int(math.floor(len(output_paths_labels) * (train_split + dev_split)))])
        test_paths_labels.extend(output_paths_labels[int(math.floor(len(output_paths_labels) * (train_split + dev_split))):])

      random.shuffle(train_paths_labels)
      print(train_paths_labels[0])

      random.shuffle(dev_paths_labels)
      print(dev_paths_labels[0])

      random.shuffle(test_paths_labels)
      print(test_paths_labels[0])

      for paths_labels, split_category in [(train_paths_labels, "train"), (dev_paths_labels, "dev"), (test_paths_labels, "test")]:
        csv_path = os.path.join(output_dir, split_category + "_examples.csv")
        with open(csv_path, "w", newline='') as csv_file:
          writer = csv.writer(csv_file)
          for (path, label) in paths_labels:
            writer.writerow([path, label])  

def stack(case_images):
  left_cc = cv2.imread(case_images[0])[...,0]
  left_mlo = cv2.imread(case_images[1])[...,0]
  right_cc = cv2.imread(case_images[2])[...,0]
  right_mlo = cv2.imread(case_images[3])[...,0]
  stack_image = np.stack((left_cc,left_mlo,right_cc,right_mlo),axis=2)
  assert stack_image.shape == tuple(list(left_cc.shape)[:2] + [4])
  return stack_image
  
def stitch(case_images):
  left_cc = cv2.imread(case_images[0])
  left_mlo = cv2.imread(case_images[1])
  left_stitch = np.concatenate((left_cc, left_mlo), axis=1)

  right_cc = cv2.imread(case_images[2])
  right_mlo = cv2.imread(case_images[3])
  right_stitch = np.concatenate((right_cc, right_mlo), axis=1)

  stitch_image = np.concatenate((left_stitch, right_stitch), axis=0)
  return stitch_image

if __name__ == '__main__':
  main()