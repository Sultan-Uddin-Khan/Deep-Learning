#import
import os
import pandas as pd
import torch #whole library
from torch.utils.data import Dataset # import standard datsets
from skimage import io

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.annotations) #25000 image
    def __getitem__(self, index): #return specific image
        img_path=os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image=io.imread(img_path)
        y_label=torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image=self.transform(image)

        return (image, y_label)

