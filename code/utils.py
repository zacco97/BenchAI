import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt 
import torch

def create_annotations(imgs_path, lbls_path, dataset_path, train=True, output_file = "train.csv", gray=True):
    if not train: 
        output_file = "val.csv"
    id = 0
    arr = []
    for img, lbl in zip(os.listdir(imgs_path), os.listdir(lbls_path)):
        im_path = os.path.join(imgs_path, img)
        lbl_path = os.path.join(lbls_path, lbl)
        
        im = cv2.imread(im_path)
        if not gray: 
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            height, width, chs = im.shape
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            height, width = im.shape
            chs = 1
        
        with open(lbl_path, "r") as f:
            labels = f.readlines()
        
        
        boxes = []
        class_id = []
        for annotation in labels:
            annotation = annotation.split(" ")
            class_id.append(int(annotation[0]))
            x, y, w, h = list(map(lambda x: np.float32(x), annotation[1:]))
            x_min, y_min = int((x- w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            boxes.append([x_min, x_max, y_min, y_max])
        
      
        arr.append({
            "id" : id, 
            "img" : os.path.join(imgs_path, img),
            "image_size": (height, width, chs),
            "lbls": boxes, 
            "class_id" : class_id
        })
        id += 1
    
    df = pd.DataFrame(arr)
    path_to_anns = os.path.join(dataset_path, output_file)
    if os.path.exists(path_to_anns):
        os.remove(path_to_anns)
    df.to_csv(path_to_anns, encoding="utf-8", index=False)
    print("Saved in: ", path_to_anns)
    

def visualize_detection(image, bbox, label):
    img = image.detach()
    img = np.array(img.view(640, 640).numpy(), dtype=np.int16)
    img_f = np.stack([img, img, img], axis=-1)
    
    x, y, w, h = list(map(lambda x:int(x.item()), bbox[0]))
    cv2.rectangle(img_f, pt1=(x, w), pt2=(y, h), color=(0, 255, 0), thickness=3)
    plt.imshow(img_f)
    plt.show()