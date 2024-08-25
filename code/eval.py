from utils import visualize_detection
import torch 
from dataset import CNNDataset
from torch.utils.data import DataLoader
import pandas as pd


if __name__ == "__main__":
    g = torch.Generator().manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    val_df = pd.read_csv("./dataset/head/val.csv")
    val_dataset = CNNDataset(annotations=val_df)
    data_val = DataLoader(val_dataset, shuffle=True, batch_size=1, generator=g)
    
    # load model
    model = torch.load('./model.pt')
    
    model.eval()
    for (im, class_ids, bboxs) in data_val:
        out = model(im.to(device))
        visualize_detection(im, out[0],  int(out[1]))