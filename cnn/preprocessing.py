import cv2
import numpy as np
import os
import glob

def read_image(folder_path):

    folder_name_list = os.listdir(folder_path)

    image_list = []
    label_list = []

    for folder_name in folder_name_list:

        path = folder_path + "/" + folder_name

        image_path_list = glob.glob(path+"/*.jpg")

        for image_path in image_path_list:

            img = cv2.imread(image_path)

            label = int(folder_name)

            img = cv2.resize(img,(32,32))

            image_list.append(img)
            label_list.append(label)


    image_list = np.array(image_list)
    label_list = np.array(label_list)

    return image_list, label_list
