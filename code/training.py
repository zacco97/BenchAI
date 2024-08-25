from CNN import CNN, loss
from utils import create_annotations, visualize_detection
from dataset import CNNDataset
import pandas as pd
from torch.utils.data import DataLoader
from torch import Generator
import torch
import torch.optim as optim
from tqdm import tqdm 

if __name__ == "__main__":
    dataset_path = "./dataset/head"
    train_img_path = "./dataset/head/images/train/"
    train_lbls_path = "./dataset/head/labels/train/"
    val_imgs_path = "./dataset/head/images/val/"
    val_lbls_path = "./dataset/head/labels/val/"
    out_path = "./dataset/head/"
    
    # use just for creating the .csv files in the folder of the dataset
    # create_annotations(train_img_path, train_lbls_path, dataset_path=dataset_path)
    # create_annotations(val_imgs_path, val_lbls_path, dataset_path=dataset_path, train=False)
    
    train_df = pd.read_csv("./dataset/head/train.csv")
    val_df = pd.read_csv("./dataset/head/val.csv")
    
    train_dataset = CNNDataset(annotations=train_df)
    val_dataset = CNNDataset(annotations=val_df)
    
    # setting device to GPU 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # setting seed
    g = Generator().manual_seed(42)
    LR = 0.001
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VAL = len(val_df)
    EPOCHS = 50
    
    
    train_dataset.plot_image()
    # plotting images
    
    data_train = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE_TRAIN, generator=g)
    data_val = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE_VAL, generator=g)
    
    
    model = CNN(in_channels=1, out_channel=1, bboxes=4, debug=False)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.to(device)
    
    # trainig model
    for epoch in tqdm(range(EPOCHS)):
        correct = 0
        val_loss = 0 
        train_loss = 0
        
        model.train()
        for (img, class_id, bboxs) in data_train:
            img = img.to(device)
            class_targets = class_id.to(device)
            bbox_targets = bboxs.to(device)

            # forward pass 
            bbox_preds, class_preds = model(img)
            # combined loss
            bbox_loss, class_loss = loss(bbox_preds, class_preds, bbox_targets, class_targets)
            combined_loss = bbox_loss + class_loss
            optimizer.zero_grad()
            
            # backward pass
            combined_loss.backward()
            
            # update 
            optimizer.step()
            
            # compute loss 
            train_loss += combined_loss.item()
           
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(data_train)}")
    
    torch.save(model, './model.pt')