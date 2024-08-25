from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
import cv2
import torch
import numpy as np
class CNNDataset(Dataset):
    def __init__(self, annotations, gray=True):
        self.annotations = annotations
        self.gray = gray
        pass
    
    def __len__(self):
        return len(self.annotations)
    
    def plot_image(self):
        num_imgs = 2
        fig, ax = plt.subplots(2, 2)
        for i in range(num_imgs):
            val = random.randint(0, self.__len__()-1)
            img = self[val][0].permute(1, 2, 0)
            
            img = np.array(img.view(640, 640).numpy(), dtype=np.int16)
            img_f = np.stack([img, img, img], axis=-1)
            
            # drawing bounding box
            bbox = self[val][2]
            x, y, w, h = list(map(lambda val: int(val.item()), bbox[0]))
            cv2.rectangle(img_f, pt1=(x, w), pt2=(y, h), color=(0, 255, 0), thickness=3)
            
            if self.gray:
                ax[i, 0].imshow(img_f)
                ax[i, 1].hist(img.ravel(), bins=range(256), fc='k', ec='k')
                
            else:
                #TODO implemet the hist and colorbar for RGB/BGR images
                plt.imshow(img)
                plt.show()
        plt.show()
    
    def __getitem__(self, index):
        ann = self.annotations.iloc[index, :]
        im_path = ann["img"]
        img_size = eval(ann["image_size"])
        bboxs = eval(ann["lbls"])
        class_ids = eval(ann["class_id"])
        
        im = cv2.imread(im_path)
        if not self.gray: 
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = torch.Tensor(im).view(int(img_size[0]), int(img_size[1]), 1).permute(2, 0, 1)
        class_ids = torch.Tensor(class_ids)
        bboxs = torch.Tensor(bboxs) 
        return (im, class_ids, bboxs)