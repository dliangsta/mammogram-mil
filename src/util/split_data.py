import os
import random
import math
import shutil
import csv

from glob import glob

random.seed(42)

def main():
  train_split, dev_split, test_split = 0.9, 0.05, 0.05
  group_cases = [True, False]
  check_overlays = [True, True]
  suffixes = ["","_augmented","_large"]
  input_dirs = ["../../data/ddsm", "../../data/augmented_data", "../../data/ddsm_large"]
  output_dirs = ["../../data/group_by_case", "../../data/shuffle_within_case"]
  label_map = {"normals": 0, "benigns": 0, "benign_without_callbacks": 0, "cancers": 1}

  assert (train_split + dev_split + test_split) == 1
  assert len(group_cases) == len(output_dirs)

  for input_dir, suffix in zip(input_dirs, suffixes):
    if suffix == "_augmented":
      continue
    for group_case, check_overlay, output_dir in zip(group_cases, check_overlays, output_dirs):
      output_dir += suffix
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)

      print(output_dir)
      train_paths_labels = []
      dev_paths_labels = []
      test_paths_labels = []
      # Loop through each classification category. Preserve the split ratio for each category.
      for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        case_dirs = os.listdir(category_dir)

        # Make sure each case stays in the same set.
        if group_case:
          case_dirs = [case_dir for case_dir in case_dirs]
          random.shuffle(case_dirs)

        # Creates list of tuples (path, label)
        output_paths_labels = []
        for case_dir in case_dirs:
          # Only keep case if there are 4 images
          if len(glob(os.path.join(category_dir, case_dir) + '/*.png')) % 4 == 0 or not group_case:
            for case_file in os.listdir(os.path.join(category_dir, case_dir)):
              if case_file.endswith(".png"):
                if category == "cancers" and case_file.replace(".png", ".OVERLAY") not in os.listdir(os.path.join(category_dir, case_dir)) and check_overlay:
                  output_paths_labels.append((os.path.join(category_dir, case_dir, case_file), label_map["normals"]))
                else:
                  if category == "cancers" or random.random() < 1/7:
                    output_paths_labels.append((os.path.join(category_dir, case_dir, case_file), label_map[category]))

        # Shuffle within each case.
        if not group_case:
          random.shuffle(output_paths_labels)

        train_end = (int(len(output_paths_labels) * train_split) // 4) * 4
        dev_end = train_end + (int(len(output_paths_labels) * dev_split) // 4) * 4
        train_paths_labels.extend(output_paths_labels[:train_end])
        dev_paths_labels.extend(output_paths_labels[train_end:dev_end])
        test_paths_labels.extend(output_paths_labels[dev_end:])

        if not group_case:
          random.shuffle(train_paths_labels)
          random.shuffle(dev_paths_labels)
          random.shuffle(test_paths_labels)

      print(train_paths_labels[0])
      print(dev_paths_labels[0])
      print(test_paths_labels[0])

      for paths_labels, split_category in [(train_paths_labels, "train"), (dev_paths_labels, "dev"), (test_paths_labels, "test")]:
        csv_path = os.path.join(output_dir, split_category + "_examples.csv")
        with open(csv_path, "w", newline='') as csv_file:
          writer = csv.writer(csv_file)
          for (path, label) in paths_labels:
            writer.writerow([path.replace('../',''), label])

      print([len(labels) for labels in [train_paths_labels, dev_paths_labels, test_paths_labels]])



if __name__ == '__main__':
  main()