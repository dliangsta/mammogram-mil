import os
path = "/home/liang/Documents/DDSM"
import imageio
import numpy as np
images = []
import matplotlib.pyplot as plt

def open():
    for root, subFolders, file_names in sorted(os.walk(path)):
        for file_name in file_names:
            if '.jpg' == os.path.splitext(file_name)[1]:
                try:
                    print(os.path.join(root, file_name))
                    # images.append(imageio.imread(os.path.join(root, file_name)) / 255.)
                    images.append(file_name)
                    # print(images[-1])
                    # print(images[-1].shape)
                    # plt.figure()
                    # plt.imshow(images[-1])
                    # plt.show()
                    # return
                except:
                    print(os.path.join(root, file_name))
                    pass


    print(len(images))

if __name__ == '__main__':
    open()